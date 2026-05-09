# ISL Bridge — Indian Sign Language Detector

> Real-time Indian Sign Language (ISL) recognition system that converts hand signs into grammatically correct English sentences using deep learning and AI.

---



## Overview

ISL Bridge is a real-time sign language recognition system built to help deaf and hard-of-hearing individuals communicate more effectively. The system detects **46 ISL classes** — full A–Z alphabet and 20 common words — through a webcam and converts detected signs into proper English sentences using the Gemini API.

---

## Features

- 🤙 Real-time hand sign detection via webcam at 30fps
- 🔤 46 ISL classes — full alphabet (A–Z) + 20 common words
- 📝 Sentence builder that accumulates detected signs into phrases
- 🤖 Gemini API converts keyword sequences into grammatically correct English
- ⚡ Sub-100ms inference latency
- 🎯 94%+ classification accuracy
- 🇮🇳 Focused on Indian Sign Language — an underrepresented sign language in existing research

---

## Tech Stack

`Python` `OpenCV` `MediaPipe` `TensorFlow/Keras` `NumPy` `scikit-learn` `Google Gemini API`

---

## How It Works

```
Webcam feed
  → MediaPipe extracts 21 hand landmarks (63 x,y,z values per frame)
  → 30-frame rolling buffer built continuously
  → Buffer fed into trained LSTM model
  → Predicted sign + confidence score displayed on screen
  → Signs accumulate into keyword sentence
  → Gemini API converts keywords → proper English sentence
  → Sentence displayed below webcam feed in real time
```

### Why Landmarks Instead of Images?

Most sign language projects train CNNs on raw images. This project uses **MediaPipe hand landmarks** instead:

- Works across all skin tones and lighting conditions
- Model size is under 5MB vs hundreds of MB for image models
- Inference is instant — no heavy image preprocessing
- Training data is 63 numbers per frame, not raw pixels

---

## Project Structure

```
Sign-Language-Detector/
├── convert_kaggle.py    ← converts Kaggle images to landmark CSV
├── collect_data.py      ← collects dynamic sign data via webcam
├── train_model.py       ← trains LSTM on collected landmarks
├── inference.py         ← runs real-time detection app
├── .env.example         ← API key template
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/abhishek3673/Sign-Language-Detector.git
cd Sign-Language-Detector
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy pandas google-generativeai python-dotenv
```

### 4. Add Gemini API key
```bash
# copy .env.example to .env and add your key
GEMINI_API_KEY=your_key_here
```
Get a free key at: [aistudio.google.com](https://aistudio.google.com)

---

## Data Collection

### Static Signs — Kaggle Dataset (A–Z, digits 1–9)
1. Download [Prathumarikeri ISL Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
2. Extract into `kaggle_data/`
3. Convert images to landmarks:
```bash
python convert_kaggle.py
```

### Dynamic Signs — Webcam Collection (J, Z + 20 words)
```bash
python collect_data.py
# enter label: J / HELLO / HELP / YES etc
# select: d (dynamic)
# press SPACE 50 times while performing the sign motion
```

**Signs to collect dynamically:**
```
Letters : J  Z
Words   : HELLO  HELP  YES  NO  SORRY  PLEASE  THANK YOU
          WATER  EAT  GOOD  BAD  NAME  WHERE  WHAT
          WHO  I  YOU  COME  GO  STOP
```

---

## Training

```bash
python train_model.py
```

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 32 |
| Optimizer | Adam |
| Architecture | 2-layer LSTM + Dense |
| Accuracy | 94%+ |
| Model size | ~5MB |
| Training time | ~15 mins (CPU) |

---

## Run

```bash
python inference.py
```

**What you see on screen:**
```
┌──────────────────────────────────┐
│                                  │
│         webcam feed              │
│      (hand landmarks drawn)      │
│                                  │
│  Sign: HELLO (94%)               │  ← current detected sign
│  HELLO WATER WHERE               │  ← keyword sentence
│  Where can I find water?         │  ← Gemini corrected sentence
└──────────────────────────────────┘
```

Press `Q` to quit.

---

## Signs Supported

**Static — from Kaggle dataset:**
```
A B C D E F G H I K L M N O P Q R S T U V W X Y
1 2 3 4 5 6 7 8 9
```

**Dynamic — self-collected:**
```
J  Z  HELLO  HELP  YES  NO  SORRY  PLEASE
THANK YOU  WATER  EAT  GOOD  BAD  NAME
WHERE  WHAT  WHO  I  YOU  COME  GO  STOP
```

---

## Results

| Metric | Value |
|--------|-------|
| Total classes | 46 |
| Training samples | ~2,300+ |
| Validation accuracy | 94%+ |
| Inference speed | <100ms |
| FPS | 30 |

---

## Social Impact

ISL Bridge is built specifically for **Indian Sign Language** — one of the most widely used yet underrepresented sign languages in existing research. The system enables deaf and hard-of-hearing individuals to communicate with hearing people without requiring them to know sign language, using only a standard webcam and no specialized hardware.

---

## Future Improvements

- [ ] Support for two-handed signs
- [ ] Mobile app version
- [ ] Expand vocabulary beyond 46 classes
- [ ] Text-to-speech for detected sentences
- [ ] Support for regional ISL variations

---

## Author

**Abhishek** — [github.com/abhishek3673](https://github.com/abhishek3673)
