import cv2
import sys

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouthCascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    ret2, frame2 = video_capture.read()
    #skala szaroœci
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    mainFace = 0
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        
        mainFace = frame[y:y+h,x:x+w]
        mainFaceGray = cv2.cvtColor(mainFace, cv2.COLOR_BGR2GRAY)
        ret2, thresh = cv2.threshold(mainFaceGray, 150, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA, offset=(x,y))

    mouths = mouthCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=40,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in mouths:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=14,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    #cv2.imshow('Video2', mainFace)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()