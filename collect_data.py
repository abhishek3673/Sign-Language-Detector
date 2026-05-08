import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = input("Enter sign label (J / HELLO / HELP etc): ").upper()
mode = input("Static or Dynamic? (s/d): ").lower()

samples = []
SAMPLES_NEEDED = 50
SEQUENCE_LEN = 30

print(f"\nPress SPACE to capture. Need {SAMPLES_NEEDED} samples for '{label}'")
print("Press Q to quit early\n")

sequence = []

while len(samples) < SAMPLES_NEEDED:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    landmarks = []
    if result.multi_hand_landmarks:
        for lm in result.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        mp_draw.draw_landmarks(frame,
                               result.multi_hand_landmarks[0],
                               mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"{label} | {len(samples)}/{SAMPLES_NEEDED}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    if mode == 'd' and landmarks:
        cv2.putText(frame, f"Frames: {len(sequence)}/{SEQUENCE_LEN}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1)

    if key == ord(' ') and landmarks:
        if mode == 's':
            repeated = landmarks * SEQUENCE_LEN
            samples.append([label] + repeated)
            print(f"Captured {len(samples)}/{SAMPLES_NEEDED}")

        elif mode == 'd':
            sequence.append(landmarks)
            print(f"Frames: {len(sequence)}/{SEQUENCE_LEN}")
            if len(sequence) == SEQUENCE_LEN:
                flat = [label] + [v for frame_lm in sequence for v in frame_lm]
                samples.append(flat)
                sequence = []
                print(f"Sequence saved! {len(samples)}/{SAMPLES_NEEDED}")

    if key == ord('q'):
        break

with open("dataset.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(samples)

print(f"\nDone! {len(samples)} samples saved for '{label}'")
cap.release()
cv2.destroyAllWindows()