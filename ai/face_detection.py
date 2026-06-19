import cv2

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # If frame not received
    if not ret:
        print("Failed to access camera")
        break

    # Show camera feed
    cv2.imshow("Smart Attendance System - Camera Test", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()