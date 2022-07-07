import cv2
from utils import anonymize_face_pixelate

def main():
    # TODO: set to False to disable pixelation
    pixelate = False
    
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
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (60,60),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        
        # save faces bounding box as list of tuple(xStart, yStart, xEnd, yEnd)
        faces_bounding = [
                (face[0], face[1], face[0] + face[2], face[1] + face[3])
                for face in faces
            ]
        
        # iterate through detected faces
        for f in faces_bounding:
            # display rectangle on detected face
            cv2.rectangle(
                frame,
                (f[0], f[1]), (f[2], f[3]),
                (0,255,0),
                2
            )
            
            if pixelate:
                # get region of interest (ROI) in frame where face is detected
                face_image = frame[f[1]:f[3], f[0]:f[2]]
                # blur ROI
                face_image_blured = anonymize_face_pixelate(face_image, 10)
                # replace ROI with blurred version
                frame[f[1]:f[3], f[0]:f[2]] = face_image_blured
            
            # display coordinates of detected face
            cv2.putText(
                frame,
                f'x: {f[0]}  y: {f[1]}', (f[0], f[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )
        
        # show 'live' video frame
        cv2.imshow('Video', frame)

        # set escape function
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()