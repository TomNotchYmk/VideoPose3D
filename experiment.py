import cv2
import numpy as np

cap = cv2.VideoCapture(0)

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

while cap.isOpened():

    ret, frame = cap.read()

    print("resolution = ", w, " x ", h)

    print("frame type = ", type(frame))

    print("shape of frame = ", frame.shape)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()