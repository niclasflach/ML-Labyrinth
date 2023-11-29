import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # convert image to hsv
        # masking for green color
    lower_red = np.array([56, 133, 24])
    upper_red = np.array([84, 255, 121])
    mask = cv.inRange(hsv, lower_red, upper_red)

    resized = cv.resize(frame[0:700, 300:1000], (200, 200))
    mask = cv.resize(mask[0:700, 300:1000], (200, 200))
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame

    cv.imshow("resized", resized)
    cv.imshow("mask", mask)

    if cv.waitKey(1) == ord("q"):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

