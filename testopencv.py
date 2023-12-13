import cv2 as cv
import numpy as np

pois = []
with open('poi.txt') as f:
    for line in f:
        line = line.strip()
        tmp = line.split(",")
        try:
            pois.append((int(tmp[0]), int(tmp[1])))
            #result.append((eval(tmp[0]), eval(tmp[1])))
        except:pass

f = open("poi.txt", "a")
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
    ball_position = cv.HoughCircles(
            mask,
            cv.HOUGH_GRADIENT,
            1.2,
            100,
            param1=100,
            param2=9,
            minRadius=2,
            maxRadius=10,
        )
    for poi in pois:
        cv.circle(resized,poi, 3, (255, 0, 0), 2 )
    if ball_position is not None:
        ball_position = np.uint16(np.around(ball_position))
        for coor in ball_position[0,:]:
            print(coor[0], coor[1])
            try:
                cv.circle(resized, (coor[0], coor[1]), coor[2], (0, 0, 255), 1 )
                noll = coor[0]
                ett = coor[1]
                post = (noll, ett)
            except:
                pass

    scale_up_x = 2
    scale_up_y = 2
    resized = cv.resize(resized, None, fx= scale_up_x, fy= scale_up_y, interpolation= cv.INTER_LINEAR)
    mask = cv.resize(mask, None, fx= scale_up_x, fy= scale_up_y, interpolation= cv.INTER_LINEAR)

    cv.imshow("resized", resized)
    cv.imshow("mask", mask)

    key = cv.waitKey(1)
    if key == ord("q"):
        f.close()
        break
    if key == ord("a"):
        pois.append(post)
        f.write(f"{noll},{ett}\n")
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

