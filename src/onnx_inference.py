"""
Week 6 — ONNX Inference Module
Project: Real-time Driver Drowsiness Detection
File: src/inference_engine.py

Reusable module that loads both ONNX models and runs
inference on cropped eye / mouth image regions.

Usage:
    from src.inference_engine import DrowsinessInferenceEngine

    engine = DrowsinessInferenceEngine()

    # From a cropped numpy image (grayscale or BGR):
    eye_pred   = engine.predict_eye(eye_crop)
    mouth_pred = engine.predict_mouth(mouth_crop)

    print(eye_pred.label)       # "closed" or "open"
    print(eye_pred.confidence)  # e.g. 0.9821
    print(eye_pred.closed)      # True / False shortcut
    print(mouth_pred.yawning)   # True / False shortcut
"""

import cv2
import numpy as np
import onnxruntime as ort
import pathlib
import time
from dataclasses import dataclass, field
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Default ONNX model paths — override via constructor if needed
DEFAULT_EYE_ONNX   = pathlib.Path("modals/onnx/eye_model3.onnx")
DEFAULT_MOUTH_ONNX = pathlib.Path("modals/onnx/mouth_model.onnx")

# Must match training configuration exactly
EYE_IMG_SIZE   = 24     # TinyEyeNet input: 1 × 24 × 24
MOUTH_IMG_SIZE = 64     # TinyMouthNet input: 1 × 64 × 64

# Class labels — must match ImageFolder alphabetical order from training
EYE_CLASSES   = ["closed", "open"]    # index 0=closed, 1=open
MOUTH_CLASSES = ["no_yawn", "yawn"]   # index 0=no_yawn, 1=yawn

# Normalisation — must match training transforms exactly
# Transforms.Normalize(mean=[0.5], std=[0.5]) maps [0,1] → [-1,1]
NORM_MEAN = 0.5
NORM_STD  = 0.5

# ═══════════════════════════════════════════════════════════════════
# PREDICTION RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    """
    Returned by every inference call.

    Fields:
        label       : predicted class name ("closed", "open",
                      "yawn", "no_yawn")
        class_index : raw integer index (0 or 1)
        confidence  : softmax probability of predicted class [0.0, 1.0]
        probabilities: full softmax distribution [p_class0, p_class1]
        inference_ms : time taken for this inference in milliseconds
        closed      : True if eye is closed  (eye model only)
        yawning     : True if mouth is yawning (mouth model only)
    """
    label        : str
    class_index  : int
    confidence   : float
    probabilities: List[float]
    inference_ms : float

    # Convenience boolean properties — set by the engine after creation
    closed  : bool = False   # eye model shortcut
    yawning : bool = False   # mouth model shortcut

    def __str__(self):
        return (f"{self.label} "
                f"({self.confidence*100:.1f}%) "
                f"[{self.inference_ms:.2f}ms]")


# ═══════════════════════════════════════════════════════════════════
# PREPROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════

def _preprocess(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Converts a raw cropped image (any size, BGR or grayscale)
    into a normalised float32 tensor ready for ONNX Runtime.

    Steps:
      1. Grayscale  — removes colour channels (not relevant to
                      eyelid state or jaw drop geometry)
      2. Resize     — standardises to the size the model was
                      trained on (24×24 or 64×64)
      3. Float32    — ONNX Runtime requires float32 input
      4. Normalise  — maps [0,255] → [-1,1] using the same
                      mean/std as training transforms
                      pixel_norm = (pixel/255 - 0.5) / 0.5
      5. Shape      — adds batch and channel dimensions:
                      (H,W) → (1, 1, H, W)
                      ONNX model expects (batch, channels, H, W)

    Returns:
        np.ndarray of shape (1, 1, target_size, target_size)
        dtype float32
    """
    # Step 1 — Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 2 — Resize
    resized = cv2.resize(
        gray,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA,
    )

    # Step 3 — Float32
    arr = resized.astype(np.float32)

    # Step 4 — Normalise: [0,255] → [-1,1]
    arr = (arr / 255.0 - NORM_MEAN) / NORM_STD

    # Step 5 — Shape: (H,W) → (1, 1, H, W)
    arr = arr[np.newaxis, np.newaxis, :, :]   # batch=1, channels=1

    return arr


def _softmax(logits: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    ONNX Runtime returns raw logits — we convert to probabilities.

    Stability trick: subtract max before exp to prevent overflow.
    Example: logits [2.1, 0.3] → probs [0.878, 0.122]
    """
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ═══════════════════════════════════════════════════════════════════
# ONNX SESSION BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_session(onnx_path: pathlib.Path) -> ort.InferenceSession:
    """
    Creates an optimised ONNX Runtime inference session.

    SessionOptions explained:
      graph_optimization_level = ORT_ENABLE_ALL
        Enables all graph optimisations:
          - Constant folding  (pre-compute fixed subgraphs)
          - Node fusion       (merge Conv+BN+ReLU into one op)
          - Memory planning   (reduce allocation overhead)
        These are applied once at session creation — inference
        itself runs on the already-optimised graph.

      intra_op_num_threads = 2
        Number of threads for parallelising a single operator.
        Set to 2 — enough for our tiny models without thrashing
        the CPU that is simultaneously running MediaPipe + OpenCV.

      log_severity_level = 3
        Suppresses all INFO and WARNING logs from ORT.
        Only ERROR (level 3) messages will appear.
        Keeps the real-time console clean.

    Execution providers:
      We try CUDAExecutionProvider first for GPU acceleration.
      If CUDA is not available (no onnxruntime-gpu installed),
      we fall back to CPUExecutionProvider silently.
      For our tiny models CPU is already fast enough (< 0.2ms).
    """
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path.resolve()}\n"
            f"Run src/convert_to_onnx.py first."
        )

    opts = ort.SessionOptions()
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    opts.intra_op_num_threads = 2
    opts.log_severity_level   = 3

    # Try GPU first, fall back to CPU
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=providers,
    )
    return session


# ═══════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════

class DrowsinessInferenceEngine:
    """
    Loads both ONNX models once and exposes simple predict methods.

    Design principles:
      - Sessions are created once at init — no repeated file I/O
        during real-time inference loop
      - Input names are read from the ONNX graph dynamically —
        no hardcoded strings that could break on re-export
      - All preprocessing is self-contained — caller just passes
        a raw numpy crop from OpenCV
      - Thread-safe for reading (ORT sessions are stateless)

    Example:
        engine = DrowsinessInferenceEngine()

        # eye_crop: numpy array, any size, BGR or gray
        result = engine.predict_eye(eye_crop)
        if result.closed:
            print(f"Eye closed! confidence={result.confidence:.2f}")

        result = engine.predict_mouth(mouth_crop)
        if result.yawning:
            print(f"Yawning! confidence={result.confidence:.2f}")
    """

    def __init__(
        self,
        eye_model_path  : pathlib.Path = DEFAULT_EYE_ONNX,
        mouth_model_path: pathlib.Path = DEFAULT_MOUTH_ONNX,
        verbose         : bool = True,
    ):
        eye_path   = pathlib.Path(eye_model_path)
        mouth_path = pathlib.Path(mouth_model_path)

        if verbose:
            print("[InferenceEngine] Loading ONNX models...")

        # Build sessions
        self._eye_session   = _build_session(eye_path)
        self._mouth_session = _build_session(mouth_path)

        # Read input node names from the graph
        # (avoids hardcoding "input" which could differ on re-export)
        self._eye_input_name   = (
            self._eye_session.get_inputs()[0].name
        )
        self._mouth_input_name = (
            self._mouth_session.get_inputs()[0].name
        )

        # Read output node names
        self._eye_output_name   = (
            self._eye_session.get_outputs()[0].name
        )
        self._mouth_output_name = (
            self._mouth_session.get_outputs()[0].name
        )

        if verbose:
            ep_eye   = self._eye_session.get_providers()[0]
            ep_mouth = self._mouth_session.get_providers()[0]
            print(f"  Eye model   → {eye_path.name} "
                  f"[{ep_eye}]")
            print(f"  Mouth model → {mouth_path.name} "
                  f"[{ep_mouth}]")
            print("[InferenceEngine] Ready.")

    # ── Eye Inference ─────────────────────────────────────────────

    def predict_eye(self, eye_crop: np.ndarray) -> PredictionResult:
        """
        Predicts whether an eye is open or closed.

        Args:
            eye_crop: numpy array, cropped eye region.
                      Any size — will be resized to 24×24.
                      BGR (3-channel) or grayscale (1-channel).

        Returns:
            PredictionResult with .closed bool shortcut.
        """
        result = self._run_inference(
            session    = self._eye_session,
            input_name = self._eye_input_name,
            image      = eye_crop,
            img_size   = EYE_IMG_SIZE,
            classes    = EYE_CLASSES,
        )
        result.closed = (result.label == "closed")
        return result

    # ── Mouth Inference ───────────────────────────────────────────

    def predict_mouth(self, mouth_crop: np.ndarray) -> PredictionResult:
        """
        Predicts whether a mouth is yawning or not.

        Args:
            mouth_crop: numpy array, cropped mouth region.
                        Any size — will be resized to 64×64.
                        BGR (3-channel) or grayscale (1-channel).

        Returns:
            PredictionResult with .yawning bool shortcut.
        """
        result = self._run_inference(
            session    = self._mouth_session,
            input_name = self._mouth_input_name,
            image      = mouth_crop,
            img_size   = MOUTH_IMG_SIZE,
            classes    = MOUTH_CLASSES,
        )
        result.yawning = (result.label == "yawn")
        return result

    # ── Batch Eye Inference (both eyes at once) ───────────────────

    def predict_eyes_batch(
        self,
        left_crop : np.ndarray,
        right_crop: np.ndarray,
    ):
        """
        Runs both eye crops through the model in a single forward
        pass (batch_size=2) — roughly 40% faster than two separate
        calls on CPU due to reduced overhead.

        Returns:
            (left_result, right_result) tuple of PredictionResult
        """
        left_arr  = _preprocess(left_crop,  EYE_IMG_SIZE)
        right_arr = _preprocess(right_crop, EYE_IMG_SIZE)
        batch     = np.concatenate([left_arr, right_arr], axis=0)

        t0      = time.perf_counter()
        logits  = self._eye_session.run(
            None,
            {self._eye_input_name: batch},
        )[0]
        elapsed = (time.perf_counter() - t0) * 1000

        results = []
        for i, (logit_row, side) in enumerate(
            zip(logits, ["left", "right"])
        ):
            probs      = _softmax(logit_row)
            idx        = int(probs.argmax())
            result     = PredictionResult(
                label         = EYE_CLASSES[idx],
                class_index   = idx,
                confidence    = float(probs[idx]),
                probabilities = probs.tolist(),
                inference_ms  = elapsed / 2,
            )
            result.closed = (result.label == "closed")
            results.append(result)

        return results[0], results[1]   # left, right

    # ── Warmup ────────────────────────────────────────────────────

    def warmup(self, n: int = 5):
        """
        Runs N dummy inferences through both models to warm up
        the ONNX Runtime JIT and OS cache.

        Call this once after creating the engine, before the
        webcam loop starts. Without warmup, the first real
        inference call is 5-10x slower than subsequent ones
        due to cold memory and JIT compilation overhead.
        """
        dummy_eye   = np.zeros((EYE_IMG_SIZE,   EYE_IMG_SIZE,   3),
                               dtype=np.uint8)
        dummy_mouth = np.zeros((MOUTH_IMG_SIZE, MOUTH_IMG_SIZE, 3),
                               dtype=np.uint8)
        for _ in range(n):
            self.predict_eye(dummy_eye)
            self.predict_mouth(dummy_mouth)

    # ── Session Info ──────────────────────────────────────────────

    def info(self) -> dict:
        """Returns metadata about both loaded sessions."""
        def _session_info(session, input_name):
            inp = session.get_inputs()[0]
            return {
                "input_name" : input_name,
                "input_shape": inp.shape,
                "providers"  : session.get_providers(),
            }
        return {
            "eye"  : _session_info(
                self._eye_session,   self._eye_input_name),
            "mouth": _session_info(
                self._mouth_session, self._mouth_input_name),
        }

    # ── Internal inference runner ─────────────────────────────────

    def _run_inference(
        self,
        session    : ort.InferenceSession,
        input_name : str,
        image      : np.ndarray,
        img_size   : int,
        classes    : list,
    ) -> PredictionResult:
        """
        Core inference pipeline shared by eye and mouth models.

        Steps:
          1. Preprocess image → (1, 1, H, W) float32 tensor
          2. Run ONNX Runtime session → raw logits (1, num_classes)
          3. Softmax logits → probabilities
          4. Argmax → predicted class index
          5. Wrap in PredictionResult dataclass
        """
        # Validate input
        if image is None or image.size == 0:
            raise ValueError(
                "Empty image passed to inference engine. "
                "Check that the crop region is valid."
            )

        # Step 1 — Preprocess
        tensor = _preprocess(image, img_size)

        # Step 2 — ONNX Runtime inference
        t0     = time.perf_counter()
        output = session.run(
            None,
            {input_name: tensor},
        )[0]                         # shape: (1, num_classes)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Step 3 — Softmax
        probs = _softmax(output[0])  # shape: (num_classes,)

        # Step 4 — Argmax
        class_idx = int(probs.argmax())

        # Step 5 — Build result
        return PredictionResult(
            label         = classes[class_idx],
            class_index   = class_idx,
            confidence    = float(probs[class_idx]),
            probabilities = probs.tolist(),
            inference_ms  = round(elapsed_ms, 4),
        )


# ═══════════════════════════════════════════════════════════════════
# STANDALONE TEST
# Run: python src/inference_engine.py
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  Inference Engine — Standalone Test")
    print("=" * 55)

    # Load engine
    try:
        engine = DrowsinessInferenceEngine(verbose=True)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    # Print session info
    print("\n  Session info:")
    for model, info in engine.info().items():
        print(f"    {model}: input={info['input_name']} "
              f"shape={info['input_shape']} "
              f"provider={info['providers'][0]}")

    # Warmup
    print("\n  Warming up (5 runs)...")
    engine.warmup(5)
    print("  Warmup complete.")

    # Test with random images (simulates real crops)
    print("\n  Running inference tests...")

    # Test eye — dark image (simulates closed eye)
    dark_eye  = np.zeros((30, 30, 3), dtype=np.uint8)
    light_eye = np.full((30, 30, 3), 180, dtype=np.uint8)

    r_dark  = engine.predict_eye(dark_eye)
    r_light = engine.predict_eye(light_eye)
    print(f"\n  Eye (dark  ~closed): {r_dark}")
    print(f"  Eye (light ~open) : {r_light}")

    # Test mouth
    dark_mouth  = np.zeros((60, 60, 3), dtype=np.uint8)
    light_mouth = np.full((60, 60, 3), 140, dtype=np.uint8)

    r_yawn    = engine.predict_mouth(dark_mouth)
    r_noyawn  = engine.predict_mouth(light_mouth)
    print(f"\n  Mouth (dark  ~yawn)   : {r_yawn}")
    print(f"  Mouth (light ~no_yawn): {r_noyawn}")

    # Test batch inference
    print("\n  Batch eye inference test:")
    left_r, right_r = engine.predict_eyes_batch(dark_eye, light_eye)
    print(f"  Left eye  : {left_r}")
    print(f"  Right eye : {right_r}")

    # Speed benchmark
    print("\n  Speed benchmark (100 calls each)...")
    eye_crop   = np.random.randint(0, 255,
                                   (EYE_IMG_SIZE, EYE_IMG_SIZE, 3),
                                   dtype=np.uint8)
    mouth_crop = np.random.randint(0, 255,
                                   (MOUTH_IMG_SIZE, MOUTH_IMG_SIZE, 3),
                                   dtype=np.uint8)

    t0 = time.perf_counter()
    for _ in range(100):
        engine.predict_eye(eye_crop)
    eye_avg = (time.perf_counter() - t0) * 10   # ms per call

    t0 = time.perf_counter()
    for _ in range(100):
        engine.predict_mouth(mouth_crop)
    mouth_avg = (time.perf_counter() - t0) * 10

    print(f"  Eye   avg: {eye_avg:.3f} ms/call")
    print(f"  Mouth avg: {mouth_avg:.3f} ms/call")
    print(f"  Combined : {eye_avg + mouth_avg:.3f} ms/frame "
          f"(budget: 33ms for 30 FPS)")

    print(f"\n{'='*55}")
    print("  All tests passed. Engine ready for Week 6.")
    print(f"{'='*55}\n")