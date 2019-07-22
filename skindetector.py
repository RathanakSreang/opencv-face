import imutils
import numpy as np
import cv2

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

camera = cv2.VideoCapture(0)
# keep looping over the frames in the video
while True:
  # grab the current frame
  (grabbed, frame) = camera.read()

  # if we are viewing a video and we did not grab a
  # frame, then we have reached the end of the video
  if not grabbed:
    break

  frame = imutils.resize(frame, height=600)
  imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
  skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

  skinYCrCb = cv2.bitwise_and(frame, frame, mask = skinRegionYCrCb)

  cv2.imshow("images", frame)
  cv2.imshow("images_skin", skinYCrCb)

  # if the 'q' key is pressed, stop the loop
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
