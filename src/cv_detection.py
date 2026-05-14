"""
Week 1 — Rule-Based Drowsiness Detection
Project: Real-time Driver Drowsiness Detection
Scope: OpenCV + MediaPipe only. No ML, no PyTorch, no ONNX.

Features:
  - EAR (Eye Aspect Ratio)      → eye closure detection
  - MAR (Mouth Aspect Ratio)    → yawn detection
  - Closure duration tracking   → sustained-closure alert
  - Visual alert overlay        → red flash on drowsiness / yawn
  - CSV logging                 → frame-by-frame EAR / MAR / alerts
"""

import csv
import time
import pathlib
import cv2
import mediapipe as mp
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# LANDMARK INDEX SETS  (MediaPipe 478-point Face Mesh)
# ═══════════════════════════════════════════════════════════════════

# Each eye: 6 points forming a vertical/horizontal cross
#   p1─────────────────p4   (horizontal: inner ↔ outer corner)
#       p2         p6
#       p3         p5       (vertical: upper ↔ lower lid pairs)
#
# EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)

LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]   # [p1,p2,p3,p4,p5,p6]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]   # [p1,p2,p3,p4,p5,p6]

# Extra ring points for drawing only
LEFT_EYE_DRAW  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE_DRAW = [33,  7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

# Mouth: 4 points
#   p1 ──────────── p3   (horizontal: left ↔ right corner)
#        p2              (upper inner lip centre)
#        p4              (lower inner lip centre)
#
# MAR = (‖p2−p4‖) / (‖p1−p3‖)

MOUTH_MAR  = [78, 81, 13, 311, 308, 402]  # [left, upper-l, upper-c, upper-r, right, lower-c]
# Simplified 4-point version used in formula: left=78, top=13, right=308, bottom=14
MOUTH_MAR_SIMPLE = [78, 13, 308, 14]       # left, top, right, bottom

MOUTH_DRAW = [61,146,91,181,84,17,314,405,321,375,291,
              61,185,40,39,37,0,267,269,270,409,291,
              78,95,88,178,87,14,317,402,318,324,308,
              78,191,80,81,82,13,312,311,310,415,308]

# ═══════════════════════════════════════════════════════════════════
# THRESHOLDS & TIMING
# ═══════════════════════════════════════════════════════════════════

EAR_THRESH          = 0.21    # below → eye considered closed
MAR_THRESH          = 0.65    # above → mouth considered open (yawn)

DROWSY_TIME_SEC     = 2.0     # seconds eyes must stay closed → DROWSY alert
YAWN_TIME_SEC       = 1.5     # seconds mouth must stay open  → YAWN alert

ALERT_DISPLAY_SEC   = 2.0     # how long the alert banner stays on screen

# ═══════════════════════════════════════════════════════════════════
# COLOURS  (BGR)
# ═══════════════════════════════════════════════════════════════════

C_EYE       = (0,   220,   0)
C_MOUTH     = (220, 220,   0)
C_ALERT_BG  = (0,    0,  200)
C_YAWN_BG   = (0,  140,  220)
C_OK_BG     = (0,  140,    0)
C_TEXT      = (255, 255, 255)
C_HUD_BG    = (20,   20,  20)
C_EAR_BAR   = (80,  200,  80)
C_MAR_BAR   = (80,  200, 220)
C_BAR_BG    = (50,   50,  50)

# ═══════════════════════════════════════════════════════════════════
# CSV SETUP
# ═══════════════════════════════════════════════════════════════════

LOG_PATH = pathlib.Path("drowsiness_log.csv")

def init_csv(path: pathlib.Path):
    writer_file = open(path, "w", newline="")
    writer = csv.writer(writer_file)
    writer.writerow([
        "timestamp",
        "frame",
        "ear_left",
        "ear_right",
        "ear_avg",
        "mar",
        "eyes_closed",
        "mouth_open",
        "closure_duration_s",
        "yawn_duration_s",
        "alert_drowsy",
        "alert_yawn",
    ])
    return writer_file, writer

# ═══════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════

def _px(landmark, idx, w, h):
    lm = landmark[idx]
    return np.array([lm.x * w, lm.y * h])


def compute_ear(landmarks, indices, w, h):
    """
    Eye Aspect Ratio — Soukupová & Čech (2016).
    indices: [p1, p2, p3, p4, p5, p6]
      p1,p4 = horizontal corners
      p2,p6 = upper lid pair
      p3,p5 = lower lid pair
    EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2·‖p1−p4‖)
    """
    p = [_px(landmarks, i, w, h) for i in indices]
    vert_a = np.linalg.norm(p[1] - p[5])
    vert_b = np.linalg.norm(p[2] - p[4])
    horiz  = np.linalg.norm(p[0] - p[3])
    return (vert_a + vert_b) / (2.0 * horiz + 1e-6)


def compute_mar(landmarks, indices, w, h):
    """
    Mouth Aspect Ratio.
    indices: [left, top, right, bottom]
    MAR = ‖top−bottom‖ / ‖left−right‖
    """
    p = [_px(landmarks, i, w, h) for i in indices]
    vert  = np.linalg.norm(p[1] - p[3])
    horiz = np.linalg.norm(p[0] - p[2])
    return vert / (horiz + 1e-6)

# ═══════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════

def draw_roi_points(frame, landmarks, indices, colour, w, h, radius=2):
    for idx in indices:
        pt = _px(landmarks, idx, w, h).astype(int)
        cv2.circle(frame, tuple(pt), radius, colour, -1, cv2.LINE_AA)


def draw_roi_contour(frame, landmarks, indices, colour, w, h):
    pts = [_px(landmarks, i, w, h).astype(int) for i in indices]
    for i in range(len(pts)):
        cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]),
                 colour, 1, cv2.LINE_AA)


def draw_metric_bar(frame, label, value, max_val, threshold,
                    bar_colour, x, y, bar_w=120, bar_h=14):
    """Horizontal bar showing a metric value vs threshold."""
    # Background
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), C_BAR_BG, -1)
    # Filled portion
    fill = int(min(value / max_val, 1.0) * bar_w)
    cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), bar_colour, -1)
    # Threshold line
    tx = x + int((threshold / max_val) * bar_w)
    cv2.line(frame, (tx, y - 2), (tx, y + bar_h + 2), (0, 0, 220), 2)
    # Label + value
    cv2.putText(frame, f"{label}: {value:.3f}", (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)


def draw_hud(frame, fps, ear_l, ear_r, ear_avg, mar,
             closure_dur, yawn_dur, alert_drowsy, alert_yawn, show_help):
    h, w = frame.shape[:2]

    # ── Top bar ──────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_TEXT, 1, cv2.LINE_AA)
    status = "DROWSY!" if alert_drowsy else ("YAWNING" if alert_yawn else "ALERT")
    s_col  = (0,0,220) if alert_drowsy else ((0,200,200) if alert_yawn else C_OK_BG)
    cv2.putText(frame, status, (w//2 - 45, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, s_col, 2, cv2.LINE_AA)

    # ── Metric bars (bottom-left panel) ──────────────────────────
    panel_x, panel_y = 10, h - 130
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (panel_x - 5, panel_y - 20),
                  (panel_x + 165, h - 10), C_HUD_BG, -1)
    cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)

    draw_metric_bar(frame, "EAR-L", ear_l, 0.5, EAR_THRESH, C_EAR_BAR,
                    panel_x, panel_y)
    draw_metric_bar(frame, "EAR-R", ear_r, 0.5, EAR_THRESH, C_EAR_BAR,
                    panel_x, panel_y + 30)
    draw_metric_bar(frame, "EAR  ", ear_avg, 0.5, EAR_THRESH, C_EAR_BAR,
                    panel_x, panel_y + 60)
    draw_metric_bar(frame, "MAR  ", mar, 1.2, MAR_THRESH, C_MAR_BAR,
                    panel_x, panel_y + 90)

    # Closure / yawn timers
    cv2.putText(frame,
                f"Closed: {closure_dur:.1f}s  Yawn: {yawn_dur:.1f}s",
                (panel_x, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_TEXT, 1, cv2.LINE_AA)

    # ── Help text ────────────────────────────────────────────────
    if show_help:
        hints = ["[M] mesh", "[H] help", "[Q] quit"]
        for i, hint in enumerate(hints):
            cv2.putText(frame, hint, (w - 110, 58 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)


def draw_alert_banner(frame, message, bg_colour):
    """Full-width flashing alert banner in the centre of the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    bh = 70
    by = h // 2 - bh // 2
    cv2.rectangle(overlay, (0, by), (w, by + bh), bg_colour, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
    tx = (w - text_size[0]) // 2
    ty = by + bh // 2 + text_size[1] // 2
    cv2.putText(frame, message, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, C_TEXT, 2, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── MediaPipe ───────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── Webcam ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── CSV ──────────────────────────────────────────────────────
    csv_file, csv_writer = init_csv(LOG_PATH)
    print(f"Logging to: {LOG_PATH.resolve()}")

    # ── State ────────────────────────────────────────────────────
    eyes_closed_since   = None   # timestamp when eyes first closed
    mouth_open_since    = None   # timestamp when mouth first opened
    alert_drowsy_until  = 0.0   # wall-clock time to keep alert visible
    alert_yawn_until    = 0.0

    show_mesh  = False
    show_help  = True
    fps        = 0.0
    ticker     = cv2.getTickCount()
    frame_idx  = 0

    print("Drowsiness detector running.")
    print("  [M] toggle mesh   [H] toggle help   [Q] quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        now   = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)
        rgb.flags.writeable = True

        # ── Default metric values (no face) ──────────────────────
        ear_l = ear_r = ear_avg = mar = 0.0
        eyes_closed = mouth_open = False
        alert_drowsy = alert_yawn = False
        closure_dur  = yawn_dur  = 0.0

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # ── Compute EAR / MAR ─────────────────────────────────
            ear_l   = compute_ear(lm, LEFT_EYE_EAR,      w, h)
            ear_r   = compute_ear(lm, RIGHT_EYE_EAR,     w, h)
            ear_avg = (ear_l + ear_r) / 2.0
            mar     = compute_mar(lm, MOUTH_MAR_SIMPLE,  w, h)

            eyes_closed = ear_avg < EAR_THRESH
            mouth_open  = mar     > MAR_THRESH

            # ── Eye closure duration ──────────────────────────────
            if eyes_closed:
                if eyes_closed_since is None:
                    eyes_closed_since = now
                closure_dur = now - eyes_closed_since
                if closure_dur >= DROWSY_TIME_SEC:
                    alert_drowsy_until = now + ALERT_DISPLAY_SEC
            else:
                eyes_closed_since = None
                closure_dur = 0.0

            # ── Yawn duration ─────────────────────────────────────
            if mouth_open:
                if mouth_open_since is None:
                    mouth_open_since = now
                yawn_dur = now - mouth_open_since
                if yawn_dur >= YAWN_TIME_SEC:
                    alert_yawn_until = now + ALERT_DISPLAY_SEC
            else:
                mouth_open_since = None
                yawn_dur = 0.0

            # ── Draw ROIs ─────────────────────────────────────────
            if show_mesh:
                for conn in mp.solutions.face_mesh.FACEMESH_TESSELATION:
                    p1 = lm[conn[0]]
                    p2 = lm[conn[1]]
                    cv2.line(frame,
                             (int(p1.x*w), int(p1.y*h)),
                             (int(p2.x*w), int(p2.y*h)),
                             (55, 55, 55), 1, cv2.LINE_AA)

            e_col = (0, 0, 220) if eyes_closed else C_EYE
            draw_roi_contour(frame, lm, LEFT_EYE_DRAW,  e_col, w, h)
            draw_roi_contour(frame, lm, RIGHT_EYE_DRAW, e_col, w, h)
            draw_roi_points (frame, lm, LEFT_EYE_EAR,   e_col, w, h, 3)
            draw_roi_points (frame, lm, RIGHT_EYE_EAR,  e_col, w, h, 3)

            m_col = (0, 140, 220) if mouth_open else C_MOUTH
            draw_roi_points(frame, lm, MOUTH_DRAW, m_col, w, h, 2)

        else:
            cv2.putText(frame, "No face detected", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2, cv2.LINE_AA)

        # ── Active alerts ─────────────────────────────────────────
        alert_drowsy = now < alert_drowsy_until
        alert_yawn   = now < alert_yawn_until

        if alert_drowsy:
            draw_alert_banner(frame, "!! DROWSINESS ALERT !!", C_ALERT_BG)
        elif alert_yawn:
            draw_alert_banner(frame, "  YAWN DETECTED  ", C_YAWN_BG)

        # ── HUD ───────────────────────────────────────────────────
        tick_now = cv2.getTickCount()
        fps      = cv2.getTickFrequency() / (tick_now - ticker + 1e-9)
        ticker   = tick_now

        draw_hud(frame, fps, ear_l, ear_r, ear_avg, mar,
                 closure_dur, yawn_dur, alert_drowsy, alert_yawn, show_help)

        # ── CSV logging ───────────────────────────────────────────
        csv_writer.writerow([
            f"{now:.4f}",
            frame_idx,
            f"{ear_l:.4f}",
            f"{ear_r:.4f}",
            f"{ear_avg:.4f}",
            f"{mar:.4f}",
            int(eyes_closed),
            int(mouth_open),
            f"{closure_dur:.4f}",
            f"{yawn_dur:.4f}",
            int(alert_drowsy),
            int(alert_yawn),
        ])
        frame_idx += 1

        cv2.imshow("Week 1 — Drowsiness Detector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mesh = not show_mesh
        elif key == ord('h'):
            show_help = not show_help

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    face_mesh.close()
    print(f"Session ended. {frame_idx} frames logged → {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()