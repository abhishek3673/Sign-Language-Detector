import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

KAGGLE_DIR = "kaggle_data"
OUTPUT_CSV = "dataset.csv"
SEQUENCE_LEN = 30
SKIP_LABELS = {'J', 'Z'}

print("Starting conversion...")
print("Found folders:", os.listdir(KAGGLE_DIR))
rows_written = 0

with open(OUTPUT_CSV, "a", newline="") as f:
    writer = csv.writer(f)

    for label in sorted(os.listdir(KAGGLE_DIR)):
        # SKIP J and Z
        if label.upper() in SKIP_LABELS:
            print(f"Skipping {label} — will collect dynamically")
            continue

        label_path = os.path.join(KAGGLE_DIR, label)
        if not os.path.isdir(label_path):
            continue

        print(f"Processing: {label}")
        count = 0

        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                landmarks = []
                for lm in result.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                repeated = landmarks * SEQUENCE_LEN
                writer.writerow([label.upper()] + repeated)
                count += 1

        print(f"  {label}: {count} samples saved")
        rows_written += count

print(f"\nDone! Total rows: {rows_written}")