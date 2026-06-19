import cv2
import os

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

student_name = input("Enter Student Name: ")

dataset_path = f"datasets/students/{student_name}"

os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while count < 30:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100,100)
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        key = cv2.waitKey(1)

        if key == ord("c"):

            face_crop = frame[y:y+h, x:x+w]

            img_path = f"{dataset_path}/{count}.jpg"

            cv2.imwrite(img_path, face_crop)

            print(f"Saved {img_path}")

            count += 1

    cv2.putText(
        frame,
        f"Captured: {count}/30",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Register Student", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Registration Complete")