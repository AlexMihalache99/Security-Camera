import cv2
import time
import datetime
import numpy as np

cap = cv2.VideoCapture(0)

#xml for eye and face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

#variables
detection = False
detection_stoppped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

#video and frame size
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    #if face detected start the camera
    if (len(faces) > 0):
        if detection:
            timer_started = False
        else:
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 24, frame_size)
            cv2.imwrite(f"{current_time}.jpg", frame)
            print("Recording started")
            
    elif detection:
        if timer_started:
            #quitting after 5 seconds of not seeing a face on the camera
            if time.time() - detection_stoppped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Recording stopped")
        else:
            timer_started = True
            detection_stoppped_time = time.time()
    
    if detection:
        out.write(frame)

    #drawing rectangles when detecting faces and eyes
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 3)
        roi_gray = gray[y:y+width, x:x+width]
        roi_color = frame[y:y+height, x:x+height]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
    
    cv2.imshow("frame", frame)

    #color detection - blue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("image", result)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()


