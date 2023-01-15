import cv2

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# capturing the frame by frame from the webcam and by default the webcam is of index 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # ret gets the true false value and frame gets the frame of the image from the video
    if ret:
        faces = classifier.detectMultiScale(frame)
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.imshow("Face detection",frame)
    key = cv2.waitKey(30) # waitkey has two words wait is for waiting and key means it is waiting for some input and the value which we are providing will be the time it will wait

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



