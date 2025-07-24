import numpy as np
import cv2 as cv


face_cascade = cv.CascadeClassifier('Haarcascade Frontal Face Default.xml')

cap = cv.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open Camera")
    exit()

while True:
    _, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)
    cv.imshow("Face Detection", img)
    key = cv.waitKey(10) # 10 millisecond
    if key == 27:
        break
cap.release()
cv.destroyAllWindows()
