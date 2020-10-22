import cv2

trained_car_data = cv2.CascadeClassifier('cars_detector.xml')
trained_pedestrian_data = cv2.CascadeClassifier('haarcascade_fullbody.xml')

#img = cv2.imread('car_image.jpg')
video = cv2.VideoCapture('pedestrians.mp4')

while True:
    (check, frame) = video.read()
    if  not check:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car_coordinates = trained_car_data.detectMultiScale(gray_frame)
    pedestrian_coordinates = trained_pedestrian_data.detectMultiScale(gray_frame)
    
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    for (x, y, w, h) in pedestrian_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow('Car and Pedestrian Tracking', frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()