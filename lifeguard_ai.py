import cv2
import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 170)

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

last_motion_time = time.time()

print("✅ Lifeguard AI Started... Press ESC to stop")

while cap.isOpened():
    
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
    if not motion_detected and time.time() - last_motion_time > 5:
        print("🚨 Possible drowning detected!")
        engine.say("Alert! Possible drowning detected!")
        engine.runAndWait()
        last_motion_time = time.time()
    elif motion_detected:
        last_motion_time = time.time()

    cv2.imshow("Lifeguard AI Camera", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(10) == 27: 
        break

cap.release()
cv2.destroyAllWindows()
