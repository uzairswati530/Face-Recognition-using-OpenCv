import cv2

# Load the pre-trained Haar Cascade for face detection
face_cap = cv2.CascadeClassifier("C:/Users/M Uzair/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalcatface.xml")

# Start video capture
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)  # Fixed color conversion
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("video_live", video_data)
    if cv2.waitKey(10) == ord("a"):
        break

video_cap.release()
cv2.destroyAllWindows()
