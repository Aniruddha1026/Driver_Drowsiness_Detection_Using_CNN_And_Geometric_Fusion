"""
Week 1 — Webcam Landmark Visualization
Project: Real-time Driver Drowsiness Detection
Scope: OpenCV + MediaPipe only. No ML, no PyTorch, no ONNX.

Draws:
  - Left eye landmarks  (green)
  - Right eye landmarks (green)
  - Mouth landmarks     (cyan)
  - Full face mesh      (dim grey, optional toggle)
"""

import cv2
import mediapipe as mp

# ─────────────────────────────────────────────
# MediaPipe Face Mesh indices for regions of interest
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
# ─────────────────────────────────────────────

# Left eye  (from the subject's perspective)
LEFT_EYE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

# Right eye (from the subject's perspective)
RIGHT_EYE = [
    33,  7,  163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

# Outer lip + inner lip combined
MOUTH = [
    # Outer lip
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    # Inner lip
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

# ─────────────────────────────────────────────
# Drawing colours  (BGR)
# ─────────────────────────────────────────────
COL_EYE        = (0, 220, 0)      # bright green
COL_MOUTH      = (220, 220, 0)    # cyan-yellow
COL_MESH       = (60, 60, 60)     # dim grey  (full mesh)
COL_INFO_BG    = (20, 20, 20)
COL_INFO_TEXT  = (200, 200, 200)
COL_LABEL      = (255, 255, 255)


def draw_landmarks_on_frame(frame, face_landmarks, show_mesh=False):
    """
    Draw eye and mouth landmarks onto `frame` in-place.
    Optionally draws the full face mesh underneath.
    """
    h, w = frame.shape[:2]

    def lm_to_px(idx):
        lm = face_landmarks.landmark[idx]
        return int(lm.x * w), int(lm.y * h)

    # ── Full mesh (optional, drawn first so ROIs appear on top) ──
    if show_mesh:
        for conn in mp.solutions.face_mesh.FACEMESH_TESSELATION:
            p1 = lm_to_px(conn[0])
            p2 = lm_to_px(conn[1])
            cv2.line(frame, p1, p2, COL_MESH, 1, cv2.LINE_AA)

    # ── Eye landmarks ──
    for idx in LEFT_EYE:
        cv2.circle(frame, lm_to_px(idx), 2, COL_EYE, -1, cv2.LINE_AA)
    for idx in RIGHT_EYE:
        cv2.circle(frame, lm_to_px(idx), 2, COL_EYE, -1, cv2.LINE_AA)

    # Connect eye contour with lines for clarity
    for ring, colour in [(LEFT_EYE, COL_EYE), (RIGHT_EYE, COL_EYE)]:
        for i in range(len(ring)):
            cv2.line(frame,
                     lm_to_px(ring[i]),
                     lm_to_px(ring[(i + 1) % len(ring)]),
                     colour, 1, cv2.LINE_AA)

    # ── Mouth landmarks ──
    for idx in MOUTH:
        cv2.circle(frame, lm_to_px(idx), 2, COL_MOUTH, -1, cv2.LINE_AA)


def draw_hud(frame, fps, show_mesh):
    """Draws a minimal HUD (FPS + keyboard hints) onto the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), COL_INFO_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_INFO_TEXT, 1, cv2.LINE_AA)

    mesh_state = "ON" if show_mesh else "OFF"
    cv2.putText(frame, f"[M] Mesh: {mesh_state}   [Q] Quit",
                (w - 260, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, COL_INFO_TEXT, 1, cv2.LINE_AA)

    # Legend bottom-left
    cv2.circle(frame, (14, h - 40), 5, COL_EYE, -1, cv2.LINE_AA)
    cv2.putText(frame, "Eyes", (24, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_LABEL, 1, cv2.LINE_AA)

    cv2.circle(frame, (80, h - 40), 5, COL_MOUTH, -1, cv2.LINE_AA)
    cv2.putText(frame, "Mouth", (90, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_LABEL, 1, cv2.LINE_AA)


def main():
    # ── MediaPipe Face Mesh setup ──
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,   # video stream mode
        max_num_faces=1,           # driver = single face
        refine_landmarks=True,     # enables iris landmarks (468 → 478 pts)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ── Webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check device index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    show_mesh = False
    fps       = 0.0
    ticker    = cv2.getTickCount()

    print("Week 1 Landmark Visualizer running.")
    print("  [M] toggle full mesh   [Q] quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — exiting.")
            break

        # Mirror so it feels like a mirror (more natural for driver cams)
        frame = cv2.flip(frame, 1)

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        if results.multi_face_landmarks:
            draw_landmarks_on_frame(frame,
                                    results.multi_face_landmarks[0],
                                    show_mesh=show_mesh)
        else:
            # No face detected — show notice
            cv2.putText(frame, "No face detected", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 220), 2, cv2.LINE_AA)

        # FPS calculation
        now   = cv2.getTickCount()
        fps   = cv2.getTickFrequency() / (now - ticker)
        ticker = now

        draw_hud(frame, fps, show_mesh)

        cv2.imshow("Week 1 — Landmark Visualizer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mesh = not show_mesh

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    print("Done.")


if __name__ == "__main__":
    main()