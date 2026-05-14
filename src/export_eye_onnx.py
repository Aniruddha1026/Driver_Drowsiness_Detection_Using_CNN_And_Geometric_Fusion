"""
export_eye_onnx.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exports ONLY the new eye model to ONNX.
Keeps the existing mouth_model.onnx untouched.

What this does:
  1. Loads modals/eye2/best_eye_model.pth
  2. Exports to models/onnx/eye_model.onnx  (overwrites old one)
  3. Parity check: PyTorch vs ONNX Runtime
  4. Speed benchmark
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
#  PATHS
# ═══════════════════════════════════════════════════════════════════

EYE_CHECKPOINT = pathlib.Path("modals/eye3/best_eye_model.pth")
ONNX_DIR       = pathlib.Path("modals/onnx")
EYE_ONNX       = ONNX_DIR / "eye_model3.onnx"

BENCHMARK_RUNS = 200

# ═══════════════════════════════════════════════════════════════════
#  MODEL  — must match training architecture exactly
#  dropout=0.5  (new model used 0.5, not 0.4)
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
    """1 × 24 × 24 → 2 classes  (closed=0, open=1)"""
    def __init__(self, num_classes=2, dropout=0.5):
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

# ═══════════════════════════════════════════════════════════════════
#  LOAD
# ═══════════════════════════════════════════════════════════════════

def load_model():
    if not EYE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {EYE_CHECKPOINT.resolve()}\n"
            "Train the model first with train_eye_modal.py")

    ckpt  = torch.load(EYE_CHECKPOINT, map_location="cpu",
                       weights_only=True)
    model = TinyEyeNet(num_classes=2, dropout=0.5)
    model.load_state_dict(ckpt["model_state"])
    model.eval()   # critical — fixes BatchNorm to running stats

    print(f"  Checkpoint : {EYE_CHECKPOINT}")
    print(f"  Epoch      : {ckpt['epoch']}")
    print(f"  Val acc    : {ckpt['val_acc']:.4f}")
    print(f"  Classes    : {ckpt['class_names']}")
    return model, ckpt

# ═══════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════

def export(model):
    dummy = torch.zeros(1, 1, 24, 24)   # (batch, channels, H, W)

    torch.onnx.export(
        model,
        dummy,
        str(EYE_ONNX),
        opset_version       = 17,
        input_names         = ["input"],
        output_names        = ["output"],
        dynamic_axes        = {
            "input" : {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        do_constant_folding = True,
        verbose             = False,
    )
    print(f"  Exported   : {EYE_ONNX}")

# ═══════════════════════════════════════════════════════════════════
#  VALIDATE
# ═══════════════════════════════════════════════════════════════════

def validate():
    proto = onnx.load(str(EYE_ONNX))
    onnx.checker.check_model(proto)
    n_nodes = len(proto.graph.node)
    n_wts   = len(proto.graph.initializer)
    size_kb = EYE_ONNX.stat().st_size / 1024
    print(f"  Graph OK   : {n_nodes} nodes | "
          f"{n_wts} weights | {size_kb:.1f} KB")

# ═══════════════════════════════════════════════════════════════════
#  PARITY CHECK
# ═══════════════════════════════════════════════════════════════════

def parity_check(model, n=50):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(EYE_ONNX),
                                sess_options=opts,
                                providers=["CPUExecutionProvider"])

    max_diff = 0.0
    misses   = 0

    with torch.no_grad():
        for _ in range(n):
            x_pt  = torch.randn(1, 1, 24, 24)
            x_np  = x_pt.numpy().astype(np.float32)
            pt    = model(x_pt).numpy()
            ort_  = sess.run(["output"], {"input": x_np})[0]
            diff  = np.abs(pt - ort_).max()
            max_diff = max(max_diff, diff)
            if pt.argmax() != ort_.argmax():
                misses += 1

    match = 100.0 * (n - misses) / n
    ok    = "✓ PASS" if match == 100.0 else "✗ FAIL"
    print(f"  Parity     : {ok} | "
          f"max_diff={max_diff:.2e} | "
          f"class_match={match:.1f}%  ({n} samples)")
    return match

# ═══════════════════════════════════════════════════════════════════
#  BENCHMARK
# ═══════════════════════════════════════════════════════════════════

def benchmark(model):
    dummy_np = np.random.randn(1, 1, 24, 24).astype(np.float32)
    dummy_pt = torch.tensor(dummy_np)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(EYE_ONNX),
                                sess_options=opts,
                                providers=["CPUExecutionProvider"])

    # Warmup
    with torch.no_grad():
        for _ in range(10): model(dummy_pt)
    for _ in range(10):
        sess.run(["output"], {"input": dummy_np})

    # PyTorch
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(BENCHMARK_RUNS): model(dummy_pt)
    pt_ms = (time.perf_counter() - t0) * 1000 / BENCHMARK_RUNS

    # ORT
    t0 = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        sess.run(["output"], {"input": dummy_np})
    ort_ms = (time.perf_counter() - t0) * 1000 / BENCHMARK_RUNS

    speedup = pt_ms / ort_ms
    fps     = 1000 / ort_ms
    print(f"  PyTorch    : {pt_ms:.3f} ms/frame")
    print(f"  ORT        : {ort_ms:.3f} ms/frame  "
          f"({speedup:.2f}x speedup | ~{fps:.0f} FPS headroom)")
    return ort_ms

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*52}")
    print(f"  Eye Model — ONNX Export")
    print(f"{'='*52}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  ONNX Runtime : {ort.__version__}")
    print()

    model, ckpt = load_model()
    export(model)
    validate()
    match   = parity_check(model)
    ort_ms  = benchmark(model)

    # Save mini report — merges with existing report if present
    report_path = ONNX_DIR / "export_report.json"
    report = {}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

    report["eye"] = {
        "checkpoint"     : str(EYE_CHECKPOINT),
        "onnx_path"      : str(EYE_ONNX),
        "val_acc"        : float(ckpt["val_acc"]),
        "input_shape"    : [1, 24, 24],
        "class_match_pct": float(match),
        "onnxrt_ms"      : round(ort_ms, 4),
        "size_kb"        : round(EYE_ONNX.stat().st_size / 1024, 1),
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'─'*52}")
    status = "✓ PASSED" if match == 100.0 else "✗ FAILED"
    print(f"  Eye model export : {status}")
    print(f"  ONNX file        : {EYE_ONNX.resolve()}")
    print(f"  Report updated   : {report_path.resolve()}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()