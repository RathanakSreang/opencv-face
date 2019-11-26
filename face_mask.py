from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import cv2
import dlib

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("[INFO] camera sensor warming up...")
vs = VideoStream(0).start()
time.sleep(2.0)
glass = cv2.imread("data/glass.png")

# loop over the frames from the video stream
while True:
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, height=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     # detect faces in the grayayscale frame
    rects = detector(gray, 0)

    # loopop over found faces
    for rect in rects:
        shape = predictor(frame, rect)
        shape = face_utils.shape_to_np(shape)

        eyeLeftSide = 0
        eyeRightSide = 0
        eyeTopSide = 0
        eyeBottomSide = 0

        for (i, (x, y)) in enumerate(shape):
            if (i + 1) == 37:
                eyeLeftSide = x - 35
            if (i + 1) == 38:
                eyeTopSide = y - 30
            if (i + 1) == 46:
                eyeRightSide = x + 35
            if (i + 1) == 48:
                eyeBottomSide = y + 30

        cv2.rectangle(frame, (eyeLeftSide, eyeTopSide), (eyeRightSide, eyeBottomSide), (0,255, 0), 3)
        width= eyeRightSide - eyeLeftSide
        if width < 0:
            width = width * -1

        # if eyeLeftSide
        fitedGlass = imutils.resize(glass, width=width)
    # show the frame
    cv2.imshow("Frame", frame)

    cv2.imshow("Glass", fitedGlass)
    key = cv2.waitKey(1) & 0xFF

     # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
