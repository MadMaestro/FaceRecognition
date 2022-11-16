# Dysponując aktualną wiedzą (nabytą chociażby w ramach zajęć z przedmiotu Grafika Komputerowa), Państwa zadaniem jest opracowanie metody dokonującej detekcji poszczególnych cech twarzy (tj. konturu twarzy, oczu oraz ust) jak również sposobu poruszania głową. Zadanie nie polega na samodzielnym opracowaniu nowatorskiego rozwiązania (chociaż jeżeli ktoś takowe uskuteczni to będzie to jedynie pozytywnie ocenione!) ale mogą Państwo skorzystać z już gotowych detektorów (do oczu i ust) - na przykład dostępnych w ramach biblioteki OpenCV. W przypadku konturu twarzy, należy skorzystać z podstawowych operacji przetwarzania obrazów (począwszy od tradycyjnej filtracji przez binaryzację aż do operacji szkieletyzacji). W odniesieniu do "siły" ruchu głową, możemy wykorzystać gotowe biblioteki do śledzenia punktów kluczowych twarzy (np. DLib, MediaPipe), które pozwolą nam na obserwację poruszania się głowy w poszczególnych momentach czasu.

# Uwaga - zadanie należy zaprezentować w real-time (nie można skorzystać z nagrań).

# Reasumując:

# - Kontur twarzy - realizacja przy użyciu metod przetwarzania i analizy obrazów wraz ze śledzeniem ruchu - interesuje nas w tym momencie jedynie informacja o "sile" ruchu (można do tego celu skorzystać z gotowych metod - np. poprzez zapisywanie informacji o punktach kluczowych w danym momencie czasu)

# - Oczy i usta - detekcja przy użyciu gotowych do użycia metod, na przykład pochodzących z OpenCV

import cv2
import sys

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
mouthCascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
eyeCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #skala szaro�ci
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #binaryzacja
    ret2, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    mouths = mouthCascade.detectMultiScale(
        gray,
        scaleFactor=1.8,
        minNeighbors=16,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in mouths:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    #cv2.imshow('BinaryVideo', thresh)


    #cv2.imshow('Video2', thresh)

    #cv2.imshow('Video3', gray)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()