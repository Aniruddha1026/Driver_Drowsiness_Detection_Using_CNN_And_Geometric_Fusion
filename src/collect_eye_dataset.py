# collect_eye_data.py
import cv2
import mediapipe as mp
import pathlib
import time

# ── Config ──────────────────────────────────────
SAVE_DIR   = pathlib.Path("data/raw/eye")
TARGET     = 600      # images per class
DELAY_SEC  = 0.08     # ~12 captures per second

# MediaPipe landmark indices for eye crops
L_RING = [362,382,381,380,374,373,390,249,
          263,466,388,387,386,385,384,398]
R_RING = [33,7,163,144,145,153,154,155,
          133,173,157,158,159,160,161,246]

def crop_eye(frame, lm, indices, w, h, pad=0.25):
    xs = [lm[i].x*w for i in indices]
    ys = [lm[i].y*h for i in indices]
    bw = max(xs)-min(xs); bh = max(ys)-min(ys)
    x0=max(0,int(min(xs)-bw*pad)); y0=max(0,int(min(ys)-bh*pad))
    x1=min(w,int(max(xs)+bw*pad)); y1=min(h,int(max(ys)+bh*pad))
    c = frame[y0:y1, x0:x1]
    return c if c.size > 0 else None

def collect(label: str):
    out = SAVE_DIR / label
    out.mkdir(parents=True, exist_ok=True)

    mp_fm = mp.solutions.face_mesh
    fm = mp_fm.FaceMesh(static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)
    count = 0
    last_save = 0

    print(f"\nCollecting: {label.upper()}")
    if label == "open":
        print("Keep your eyes OPEN naturally. Press SPACE to start.")
    else:
        print("Keep your eyes CLOSED. Press SPACE to start.")
    print("Press Q to stop early.\n")

    # Wait for spacebar to start
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Press SPACE to start collecting {label}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)
        cv2.imshow("Eye Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): break

    while count < TARGET:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = fm.process(rgb)

        now = time.time()
        saved_this_frame = False

        if res.multi_face_landmarks and (now - last_save) >= DELAY_SEC:
            lm = res.multi_face_landmarks[0].landmark

            for eye_indices in [L_RING, R_RING]:
                crop = crop_eye(frame, lm, eye_indices, w, h)
                if crop is not None and crop.size > 0:
                    gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (24, 24))

                    # CLAHE — equalises brightness so dark rooms still produce usable images
                    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                    resized = clahe.apply(resized)

                    filename = out / f"{label}_{count:05d}.png"
                    cv2.imwrite(str(filename), resized)
                    count += 1
                    saved_this_frame = True
                    if count >= TARGET: break

            if saved_this_frame:
                last_save = now

        # HUD
        progress = int((count / TARGET) * 300)
        cv2.rectangle(frame, (20, h-50), (320, h-20), (50,50,50), -1)
        cv2.rectangle(frame, (20, h-50), (20+progress, h-20), (0,200,0), -1)
        cv2.putText(frame, f"{label.upper()} : {count}/{TARGET}",
                    (20, h-55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        if res.multi_face_landmarks:
            cv2.putText(frame, "Face detected", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.putText(frame, "No face - move closer",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 2)

        cv2.imshow("Eye Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    fm.close()
    cv2.destroyAllWindows()
    print(f"  Saved {count} {label} eye images → {out.resolve()}")
    return count

# ── Main ─────────────────────────────────────────
if __name__ == "__main__":
    print("="*50)
    print("  Eye Data Collector — Webcam Style")
    print("="*50)
    print(f"\nWill collect {TARGET} images per class.")
    print("Both LEFT and RIGHT eye are captured each frame.")
    print("Estimated time: ~3-4 minutes per class.\n")

    n_open   = collect("open")
    n_closed = collect("closed")

    print(f"\n{'='*50}")
    print(f"  Done!")
    print(f"  open  : {n_open}  images")
    print(f"  closed: {n_closed} images")
    print(f"  Run eye_data_prep.py next.")
    print(f"{'='*50}")