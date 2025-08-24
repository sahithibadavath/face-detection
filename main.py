import cv2

harcascade_path = 'model/haarcascade_frontalface_default.xml'
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height
while True:
    success, imag= cap.read()
    facecascade = cv2.CascadeClassifier(r"C:\Users\badav\OneDrive\Desktop\face detection\model\haarcascade_frontalface_default.xml")

    imag_gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    face = facecascade.detectMultiScale(imag_gray, 1.1, 4)

    for (x, y, w, h) in face:
        cv2.rectangle(imag, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(imag, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Image", imag)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break