import os
import cv2

# pre-trained face classifier path
cascPath = './assets/haarcascade_frontalface_alt2.xml'

# load pre-trained face classifier
faceCascade = cv2.CascadeClassifier(cascPath)

# init video capture
video_capture = cv2.VideoCapture(0)

while True:
    # capture video
    ret, frame = video_capture.read()

    # convert to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect objects using face classifier
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor = 1.1,
                                         minNeighbors = 5,
                                         minSize = (60,60),
                                         flags = cv2.CASCADE_SCALE_IMAGE)
    
    # draw rectangle on detected face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    
    # show coordinates of detected face
    for (x,y,w,h) in faces:
        cv2.putText(frame, f'x: {x}  y: {y}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    # show 'live' video frame
    cv2.imshow('Video', frame)

    # set escape function
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
video_capture.release()
cv2.destroyAllWindows()