# check_webcam_brightness.py
import cv2
cap = cv2.VideoCapture(0)
print("Press Q when happy with lighting.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()

    colour = (0,200,0) if 80 <= mean <= 180 else (0,0,220)
    label  = "GOOD" if 80 <= mean <= 180 else "TOO DARK - ADD LIGHT"

    cv2.putText(frame, f"Brightness: {mean:.0f}  {label}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, colour, 2)
    cv2.imshow("Brightness Check", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()