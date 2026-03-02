import os
import cv2
from deepface import DeepFace

# =========================
# 1. SETTINGS
DATASET_PATH = r"C:\Users\hp\Desktop\DeepFace\archive\lfw-funneled\lfw_funneled"
MODEL_NAME = "Facenet"
LIMIT_IMAGES = 5  # Set to None to load all images

# =========================
# 2. FUNCTIONS

def gather_images(dataset_path, limit=None):
    """Recursively gathers image file paths from the dataset."""
    images = []
    print(f"\n[INFO] Scanning: {dataset_path}")

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Path does not exist: {dataset_path}")
        return []

    count = 0
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                full_path = os.path.join(root, f)
                images.append(full_path)
                print(f"  -> [FOUND] {f}")
                count += 1
                if limit and count >= limit:
                    print(f"\n[INFO] Limit of {limit} images reached.")
                    return images
    return images


def detect_face(image_path):
    """Visual detection with OpenCV (draw rectangles)."""
    img = cv2.imread(image_path)
    if img is None: 
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(20)


def recognize_face(img1_path, img2_path):
    """Compare two images using DeepFace and print results."""
    try:
        result = DeepFace.verify(
            img1_path, img2_path, model_name=MODEL_NAME,
            distance_metric='cosine', enforce_detection=False
        )
        name1 = os.path.basename(img1_path)
        name2 = os.path.basename(img2_path)
        dist = result['distance']

        if result['verified']:
            print(f"[MATCH]    {name1} <-> {name2} | Dist: {dist:.4f}")
        else:
            print(f"[NO MATCH] {name1} vs {name2} | Dist: {dist:.4f}")
    except Exception as e:
        print(f"[ERROR] {e}")


# =========================
# 3. MAIN EXECUTION

if __name__ == "__main__":
    print("=== Facial Recognition Pipeline Started ===")

    # STEP 1: Load images
    all_images = gather_images(DATASET_PATH, LIMIT_IMAGES)
    if not all_images:
        exit()

    # STEP 2: Visual face detection (optional)
    print("\n--- Phase 1: Visual Face Detection ---")
    for img_path in all_images:
        detect_face(img_path)
    cv2.destroyAllWindows()

    # STEP 3: Load DeepFace model
    print("\n--- Phase 2: DeepFace Verification ---")
    print("[WAIT] Loading Facenet model...")
    DeepFace.build_model(MODEL_NAME)
    print("[OK] Model loaded!\n")

    # STEP 4: Choose comparison mode
    mode = input("Choose mode: [1] Compare to one reference, [2] Compare all pairwise: ")

    if mode == "1":
        # Let user select a reference image
        print("\nAvailable images:")
        for i, img in enumerate(all_images):
            print(f"{i}: {os.path.basename(img)}")
        choice = int(input("\nEnter index of reference image: "))
        reference_img = all_images[choice]

        print(f"\n[INFO] Comparing everyone against: {os.path.basename(reference_img)}\n")
        for i, img_path in enumerate(all_images):
            if img_path != reference_img:
                recognize_face(reference_img, img_path)

    elif mode == "2":
        # Compare every image against every other image
        print("\n[INFO] Performing full pairwise comparisons...\n")
        for i in range(len(all_images)):
            for j in range(i + 1, len(all_images)):
                recognize_face(all_images[i], all_images[j])

    print("\n[INFO] Demonstration Complete.")
