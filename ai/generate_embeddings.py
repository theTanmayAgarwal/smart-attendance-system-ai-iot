import os
import cv2
import numpy as np

from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

DATASET_PATH = "datasets/students"
EMBEDDING_PATH = "database/embeddings"

os.makedirs(EMBEDDING_PATH, exist_ok=True)

for student_name in os.listdir(DATASET_PATH):

    student_folder = os.path.join(DATASET_PATH, student_name)

    if not os.path.isdir(student_folder):
        continue

    embeddings = []

    print(f"\nProcessing {student_name}...")

    for image_name in os.listdir(student_folder):

        image_path = os.path.join(student_folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        faces = app.get(image)

        if len(faces) == 0:
            continue

        embedding = faces[0].embedding

        embeddings.append(embedding)

    if len(embeddings) == 0:
        print(f"No valid faces found for {student_name}")
        continue

    mean_embedding = np.mean(embeddings, axis=0)

    save_path = os.path.join(
        EMBEDDING_PATH,
        f"{student_name.lower()}.npy"
    )

    np.save(save_path, mean_embedding)

    print(f"Saved: {save_path}")

print("\nEmbedding generation complete.")