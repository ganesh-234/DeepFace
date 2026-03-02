from deepface import DeepFace
import cv2
import os

# Configuration
DATASET_PATH = r"C:\Users\hp\Desktop\DeepFace\dataset"
MODEL_NAME = "Facenet"

# Detect face function
def detect_face(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Image not found: {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Detected Face", img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    print(f"[INFO] Faces detected in {os.path.basename(image_path)}: {len(faces)}")
    return len(faces) > 0

# Facial recognition function
def recognize_face(img1_path, img2_path):
    result = DeepFace.verify(
    img1_path,
    img2_path,
    model_name=MODEL_NAME,
    distance_metric='cosine',
    enforce_detection=False
)

    print(f"[INFO] {os.path.basename(img1_path)} vs {os.path.basename(img2_path)} | Verified: {result['verified']} | Distance: {result['distance']:.4f}")
    return result['verified']

# Gather all images recursively
def gather_images(dataset_path):
    images = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".jpg", ".png")):
                images.append(os.path.join(root, f))
    return images

# Main
if __name__ == "__main__":
    print("=== Facial Recognition Project ===")

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset path not found: {DATASET_PATH}")
        exit()

    all_images = gather_images(DATASET_PATH)

    if len(all_images) < 2:
        print("[WARNING] Please add at least two images (.jpg or .png) to the dataset or its subfolders.")
    else:
        print(f"[INFO] Found {len(all_images)} images for analysis.")

        # Detect faces in all images
        for img_path in all_images:
            detect_face(img_path)

        # Compare every image with every other image
        for i in range(len(all_images)):
            for j in range(i + 1, len(all_images)):
                recognize_face(all_images[i], all_images[j])
