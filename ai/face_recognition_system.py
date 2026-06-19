import cv2
import os
import numpy as np

from insightface.app import FaceAnalysis

# ----------------------------
# Initialize InsightFace
# ----------------------------

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

# ----------------------------
# Load Known Embeddings
# ----------------------------

known_faces = {}

embedding_folder = "database/embeddings"

for file in os.listdir(embedding_folder):

    if file.endswith(".npy"):

        name = file.replace(".npy", "")

        embedding = np.load(
            os.path.join(embedding_folder, file)
        )

        known_faces[name] = embedding

print("Loaded Faces:")
print(known_faces.keys())

# ----------------------------
# Webcam
# ----------------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    faces = app.get(frame)

    for face in faces:

        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox

        current_embedding = face.embedding

        best_match = "Unknown"
        best_score = float("inf")

        for name, saved_embedding in known_faces.items():

            distance = np.linalg.norm(
                current_embedding - saved_embedding
            )

            if distance < best_score:

                best_score = distance
                best_match = name

        # Threshold
        if best_score > 18:
            best_match = "Unknown"

        # Draw Box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Display Name
        cv2.putText(
            frame,
            best_match.upper(),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Display Distance
        cv2.putText(
            frame,
            f"{best_score:.2f}",
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Smart Attendance - Face Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()