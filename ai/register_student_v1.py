import cv2
import os

student_name = input("Enter Student Name: ")

dataset_path = f"datasets/students/{student_name}"

os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

while count < 30:

    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Register Student", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):

        img_path = f"{dataset_path}/{count}.jpg"

        cv2.imwrite(img_path, frame)

        print(f"Saved {img_path}")

        count += 1

cap.release()
cv2.destroyAllWindows()

print("Registration Complete")