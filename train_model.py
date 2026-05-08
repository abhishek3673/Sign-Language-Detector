import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

SEQUENCE_LEN = 30

# load data
df = pd.read_csv("dataset.csv", header=None)
labels = df.iloc[:, 0].values
data = df.iloc[:, 1:].values

# encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_cat = to_categorical(labels_encoded)

# save labels
with open("labels.txt", "w") as f:
    f.write("\n".join(le.classes_))

# reshape → (samples, 30, 63)
X = data.reshape(-1, SEQUENCE_LEN, 63)
y = labels_cat

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LEN, 63)),
    Dropout(0.3),
    LSTM(128),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(X_train, y_train,
          epochs=30,
          batch_size=32,
          validation_data=(X_test, y_test))

model.save("model/lstm_model.h5")
print("\nDone! Classes:", list(le.classes_))