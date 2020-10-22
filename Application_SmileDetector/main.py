import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    if not check:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(gray_frame)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        face_frame = frame[y:y+h, x:x+w]
        face_gray_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        smile_coordinates = trained_smile_data.detectMultiScale(face_gray_frame, scaleFactor = 1.7, minNeighbors= 20 )
        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'similing', (x, y+h+40), fontScale = 3, fontFace = cv2.FONT_HERSHEY_PLAIN, color = (255, 255, 255))
    cv2.imshow('Smile Detector', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()