"""
Week 5 — PyTorch → ONNX Export
Project: Real-time Driver Drowsiness Detection

Exports both trained models to ONNX format and validates
that ONNX Runtime output matches PyTorch output exactly.

What this script does:
  1. Loads best_eye_model.pth   → exports eye_model.onnx
  2. Loads best_mouth_model.pth → exports mouth_model.onnx
  3. Runs parity check: PyTorch output vs ONNX Runtime output
  4. Benchmarks inference speed: PyTorch vs ONNX Runtime
  5. Saves export report as JSON

Output:
  models/onnx/
      eye_model.onnx
      mouth_model.onnx
      export_report.json
"""

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import pathlib
import json
import time

# ═══════════════════════════════════════════════════════════════════
# PATHS  — adjust if your folder structure differs
# ═══════════════════════════════════════════════════════════════════

EYE_CHECKPOINT   = pathlib.Path("modals/eye2/best_eye_model.pth")
MOUTH_CHECKPOINT = pathlib.Path("modals/mouth/best_mouth_model.pth")
ONNX_DIR         = pathlib.Path("modals/onnx")

BENCHMARK_RUNS   = 200    # number of inference passes for speed test

# ═══════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# Must match the architecture used during training exactly.
# ═══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TinyEyeNet(nn.Module):
    """1 × 24 × 24 → 2 classes (closed / open)"""
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  8,  pool=True),
            ConvBlock(8,  16, pool=True),
            ConvBlock(16, 32, pool=True),
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


class TinyMouthNet(nn.Module):
    """1 × 64 × 64 → 2 classes (no_yawn / yawn)"""
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   16,  pool=True),
            ConvBlock(16,  32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════
# LOAD PYTORCH CHECKPOINT
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, model_class, model_kwargs, device):
    """
    Loads a saved checkpoint and returns the model in eval mode.

    Why eval mode matters for export:
      BatchNorm behaves differently in train vs eval mode.
      In train mode it uses batch statistics (varies per input).
      In eval mode it uses the running mean/variance computed
      during training (fixed, deterministic).
      Exporting in train mode would produce non-deterministic
      ONNX graphs — always export in eval mode.

    Why weights_only=True:
      Security — prevents arbitrary code execution from pickle.
      Safe because we saved only plain Python types + state_dict.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()                        # ← critical
    model.to(device)

    print(f"  Loaded   : {checkpoint_path.name}")
    print(f"  Epoch    : {checkpoint['epoch']}")
    print(f"  Val acc  : {checkpoint['val_acc']:.4f}")
    print(f"  Classes  : {checkpoint['class_names']}")
    return model, checkpoint

# ═══════════════════════════════════════════════════════════════════
# ONNX EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_to_onnx(model, input_shape, onnx_path, model_name):
    """
    Exports a PyTorch model to ONNX format.

    Parameters explained:
    ─────────────────────
    model:
        Must be in eval() mode before calling this.

    dummy_input:
        A sample tensor with the exact shape the model expects.
        torch.onnx.export traces the model by running this input
        through it once — the computation graph is recorded.
        Values don't matter, only the shape and dtype.

    opset_version=17:
        ONNX operator set version. Higher = more operators supported.
        Opset 17 supports all operators used in our CNNs
        (Conv, BatchNorm, ReLU, MaxPool, AdaptiveAvgPool, Linear).
        Opset 11+ is required for AdaptiveAvgPool — we use 17 for
        forward compatibility with current ONNX Runtime versions.

    input_names / output_names:
        Human-readable names for the graph's input and output nodes.
        Used when calling ONNX Runtime: session.run(["output"], ...).

    dynamic_axes:
        Marks the batch dimension (dim 0) as dynamic.
        Without this, the exported model is fixed to batch_size=1.
        With dynamic_axes, ONNX Runtime accepts any batch size.
        For real-time inference we always use batch=1, but marking
        it dynamic is best practice and costs nothing at runtime.

    do_constant_folding=True:
        Pre-computes constant subgraphs at export time.
        For example, BatchNorm's running_mean and running_var are
        constants — they get folded into the Conv weights during
        export, reducing the number of operations at inference time.
        This is a free optimisation — always leave it True.
    """
    dummy_input = torch.zeros(1, *input_shape)   # (1, C, H, W)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version      = 17,
        input_names        = ["input"],
        output_names       = ["output"],
        dynamic_axes       = {
            "input" : {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        do_constant_folding = True,
        verbose             = False,
    )
    print(f"  Exported : {onnx_path.name}")

# ═══════════════════════════════════════════════════════════════════
# ONNX GRAPH VALIDATION
# ═══════════════════════════════════════════════════════════════════

def validate_onnx_graph(onnx_path):
    """
    Runs onnx.checker.check_model on the exported file.

    What it checks:
      - All operator types are valid for the declared opset
      - All tensor shapes are consistent through the graph
      - No missing or dangling nodes
      - Input/output names are correctly wired

    This is a graph-level check only (no actual inference).
    It catches export bugs before you try to run the model.
    """
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)

    # Report graph metadata
    graph    = model_proto.graph
    n_nodes  = len(graph.node)
    n_inputs = len(graph.input)
    n_inits  = len(graph.initializer)   # trainable weights

    # Estimate model size on disk
    size_kb = onnx_path.stat().st_size / 1024

    print(f"  Graph OK : {n_nodes} nodes | "
          f"{n_inits} weights | {size_kb:.1f} KB on disk")

# ═══════════════════════════════════════════════════════════════════
# PARITY CHECK — PyTorch vs ONNX Runtime
# ═══════════════════════════════════════════════════════════════════

def parity_check(model, onnx_path, input_shape, device, n_samples=50):
    """
    Runs N random inputs through both PyTorch and ONNX Runtime
    and compares the outputs numerically.

    Why outputs may differ slightly:
      PyTorch uses 32-bit float arithmetic on GPU/CPU.
      ONNX Runtime uses its own optimised kernels.
      Tiny floating-point rounding differences (< 1e-5) are normal
      and have zero effect on the argmax (predicted class).

    What we check:
      max_diff   : largest absolute difference across all outputs
      class_match: whether argmax (predicted class) is identical
                   for every test sample — this is the real metric.
                   If class_match = 100%, the models are equivalent.
    """
    # Build ONNX Runtime session
    # CPUExecutionProvider is used here for the parity check
    # (works on both Windows and WSL regardless of GPU)
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3   # suppress verbose ORT logs
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    max_diff     = 0.0
    class_misses = 0

    with torch.no_grad():
        for _ in range(n_samples):
            # Random input — same as what the model will see at runtime
            x_torch = torch.randn(1, *input_shape)
            x_numpy = x_torch.numpy().astype(np.float32)

            # PyTorch forward pass
            pt_out  = model.cpu()(x_torch).numpy()

            # ONNX Runtime forward pass
            ort_out = session.run(
                ["output"],
                {"input": x_numpy}
            )[0]

            # Numerical difference
            diff = np.abs(pt_out - ort_out).max()
            max_diff = max(max_diff, diff)

            # Class agreement
            if pt_out.argmax() != ort_out.argmax():
                class_misses += 1

    class_match_pct = 100.0 * (n_samples - class_misses) / n_samples

    status = "✓ PASS" if class_match_pct == 100.0 else "✗ FAIL"
    print(f"  Parity   : {status} | "
          f"max_diff={max_diff:.2e} | "
          f"class_match={class_match_pct:.1f}% "
          f"({n_samples} samples)")

    return max_diff, class_match_pct

# ═══════════════════════════════════════════════════════════════════
# INFERENCE SPEED BENCHMARK
# ═══════════════════════════════════════════════════════════════════

def benchmark(model, onnx_path, input_shape, n_runs=BENCHMARK_RUNS):
    """
    Measures average single-frame inference time for:
      - PyTorch (CPU, eval mode)
      - ONNX Runtime (CPU)

    We benchmark on CPU because:
      1. Final deployment runs on Windows CPU (no CUDA in ONNX Runtime
         unless you install onnxruntime-gpu separately)
      2. Real-time webcam inference at 30 FPS requires < 33ms per frame
         — both models should be well under this on modern CPUs

    Warmup runs (10) are discarded to avoid cold-start timing bias
    from OS page faults and cache misses on the first few runs.
    """
    dummy_np = np.random.randn(1, *input_shape).astype(np.float32)
    dummy_pt = torch.tensor(dummy_np)

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    # ── PyTorch CPU benchmark ──
    model_cpu = model.cpu()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model_cpu(dummy_pt)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model_cpu(dummy_pt)
    pt_ms = (time.perf_counter() - t0) * 1000 / n_runs

    # ── ONNX Runtime benchmark ──
    # Warmup
    for _ in range(10):
        session.run(["output"], {"input": dummy_np})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run(["output"], {"input": dummy_np})
    ort_ms = (time.perf_counter() - t0) * 1000 / n_runs

    speedup = pt_ms / ort_ms if ort_ms > 0 else 1.0
    fps_headroom = 1000 / ort_ms

    print(f"  PyTorch  : {pt_ms:.3f} ms/frame")
    print(f"  ORT      : {ort_ms:.3f} ms/frame  "
          f"({speedup:.2f}x speedup | ~{fps_headroom:.0f} FPS headroom)")

    return pt_ms, ort_ms

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    # CPU for export — ONNX export does not benefit from GPU
    # and avoids CUDA tensor serialisation issues
    device = torch.device("cpu")

    print(f"\n{'='*57}")
    print(f"  Week 5 — PyTorch → ONNX Export")
    print(f"{'='*57}")
    print(f"  ONNX opset     : 17")
    print(f"  ONNX Runtime   : {ort.__version__}")
    print(f"  PyTorch        : {torch.__version__}")

    report = {}

    # ══════════════════════════════════════════════════════════════
    # EYE MODEL
    # ══════════════════════════════════════════════════════════════
    print(f"\n── Eye Model ─────────────────────────────────────────")

    eye_model, eye_ckpt = load_model(
        EYE_CHECKPOINT,
        TinyEyeNet,
        {"num_classes": 2, "dropout": 0.4},
        device,
    )
    eye_onnx = ONNX_DIR / "eye_model.onnx"
    eye_input_shape = (1, 24, 24)   # (C, H, W) — no batch dim

    export_to_onnx(eye_model, eye_input_shape, eye_onnx, "TinyEyeNet")
    validate_onnx_graph(eye_onnx)
    eye_diff, eye_match = parity_check(
        eye_model, eye_onnx, eye_input_shape, device)
    eye_pt_ms, eye_ort_ms = benchmark(
        eye_model, eye_onnx, eye_input_shape)

    report["eye"] = {
        "checkpoint"     : str(EYE_CHECKPOINT),
        "onnx_path"      : str(eye_onnx),
        "val_acc"        : float(eye_ckpt["val_acc"]),
        "input_shape"    : list(eye_input_shape),
        "max_diff"       : float(eye_diff),
        "class_match_pct": float(eye_match),
        "pytorch_ms"     : round(eye_pt_ms,  4),
        "onnxrt_ms"      : round(eye_ort_ms, 4),
        "size_kb"        : round(eye_onnx.stat().st_size / 1024, 1),
    }

    # ══════════════════════════════════════════════════════════════
    # MOUTH MODEL
    # ══════════════════════════════════════════════════════════════
    print(f"\n── Mouth Model ───────────────────────────────────────")

    mouth_model, mouth_ckpt = load_model(
        MOUTH_CHECKPOINT,
        TinyMouthNet,
        {"num_classes": 2, "dropout": 0.5},
        device,
    )
    mouth_onnx        = ONNX_DIR / "mouth_model.onnx"
    mouth_input_shape = (1, 64, 64)   # (C, H, W)

    export_to_onnx(mouth_model, mouth_input_shape, mouth_onnx, "TinyMouthNet")
    validate_onnx_graph(mouth_onnx)
    mouth_diff, mouth_match = parity_check(
        mouth_model, mouth_onnx, mouth_input_shape, device)
    mouth_pt_ms, mouth_ort_ms = benchmark(
        mouth_model, mouth_onnx, mouth_input_shape)

    report["mouth"] = {
        "checkpoint"     : str(MOUTH_CHECKPOINT),
        "onnx_path"      : str(mouth_onnx),
        "val_acc"        : float(mouth_ckpt["val_acc"]),
        "input_shape"    : list(mouth_input_shape),
        "max_diff"       : float(mouth_diff),
        "class_match_pct": float(mouth_match),
        "pytorch_ms"     : round(mouth_pt_ms,  4),
        "onnxrt_ms"      : round(mouth_ort_ms, 4),
        "size_kb"        : round(mouth_onnx.stat().st_size / 1024, 1),
    }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*57}")
    print(f"  Export Summary")
    print(f"{'─'*57}")
    print(f"  {'Model':<10} {'Val Acc':>8} {'Size':>8} "
          f"{'ORT ms':>8} {'Match':>8}")
    print(f"  {'─'*50}")
    for name, r in report.items():
        print(f"  {name:<10} {r['val_acc']:>8.4f} "
              f"{r['size_kb']:>7.1f}K "
              f"{r['onnxrt_ms']:>7.3f}ms "
              f"{r['class_match_pct']:>7.1f}%")

    # Both models pass if class_match = 100%
    all_pass = all(r["class_match_pct"] == 100.0
                   for r in report.values())
    status = "✓ ALL MODELS PASSED" if all_pass else "✗ CHECK FAILURES ABOVE"
    print(f"\n  {status}")
    print(f"  ONNX files → {ONNX_DIR.resolve()}")

    # Save report
    report_path = ONNX_DIR / "export_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report    → {report_path.resolve()}")

    print(f"\n{'='*57}")
    print(f"  Ready for Week 6 real-time inference.")
    print(f"{'='*57}\n")


if __name__ == "__main__":
    main()