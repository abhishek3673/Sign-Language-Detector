import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from google import genai
from dotenv import load_dotenv
import os
import time


load_dotenv()
client = genai.Client(api_key=os.getenv("AIzaSyBhkZQknGUrIGT2o7hNPEg8XbajTwG9Zv0"))


model = load_model("model/lstm_model.h5")
with open("labels.txt") as f:
    classes = f.read().splitlines()

SEQUENCE_LEN = 30
THRESHOLD = 0.85

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sequence = []
sentence = []
last_word = ""
proper_sentence = ""



def fix_grammar(keywords):
    try:
        prompt = f"Convert these sign language keywords into one proper English sentence. Return only the sentence: {' '.join(keywords)}"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return " ".join(keywords)  # fallback = just show keywords

print("Running... Press Q to quit")

while True:
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

    predicted_word = ""
    confidence = 0

    if landmarks:
        sequence.append(landmarks)
        sequence = sequence[-SEQUENCE_LEN:]

        if len(sequence) == SEQUENCE_LEN:
            X = np.expand_dims(sequence, axis=0)
            pred = model.predict(X, verbose=0)[0]
            confidence = np.max(pred)
            predicted_word = classes[np.argmax(pred)]

            if confidence > THRESHOLD and predicted_word != last_word:
                sentence.append(predicted_word)
                last_word = predicted_word

                if len(sentence) % 5 == 0:
                    proper_sentence = fix_grammar(sentence)

    h, w, _ = frame.shape
    cv2.putText(frame, f"Sign: {predicted_word} ({confidence:.0%})",
                (10, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, " ".join(sentence[-6:]),
                (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(frame, proper_sentence,
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,100), 2)

    cv2.imshow("ISL Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()