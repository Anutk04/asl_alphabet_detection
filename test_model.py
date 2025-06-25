import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# Load trained model and label encoder
with open('asl_mediapipe_model.pkl', 'rb') as f:
    model, le = pickle.load(f)

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Test image folder
test_folder = 'asl_alphabet_test'  # your flat test image folder
X_test, y_test = [], []
skipped_files = []

# Helper to get label from filename
def extract_label(filename):
    match = re.match(r'^([A-Za-z])', filename)  # A.jpg, B_2.jpg → A, B
    return match.group(1).upper() if match else None

for filename in os.listdir(test_folder):
    label = extract_label(filename)
    if not label:
        continue

    img_path = os.path.join(test_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        skipped_files.append(filename)
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        row = [pt for lm in hand.landmark for pt in (lm.x, lm.y, lm.z)]
        if len(row) == 63:
            X_test.append(row)
            y_test.append(label)
        else:
            skipped_files.append(filename)
    else:
        skipped_files.append(filename)

# Convert to NumPy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Encode labels
y_encoded = le.transform(y_test)
y_pred = model.predict(X_test)

# Classes that were actually present in test
unique_classes = np.unique(y_encoded)

# Summary
print(f"\n🧪 Total test images attempted: {len(os.listdir(test_folder))}")
print(f"✅ Successfully processed: {len(X_test)}")
print(f"❌ Skipped (no hand detected): {len(skipped_files)}")
if skipped_files:
    print("   Skipped files:", skipped_files)

# Accuracy
print("\n✅ Test Accuracy:", accuracy_score(y_encoded, y_pred))

# Classification Report (only valid classes)
print("\n📋 Classification Report:")
print(classification_report(
    y_encoded,
    y_pred,
    labels=unique_classes,
    target_names=le.inverse_transform(unique_classes)
))

# Confusion Matrix
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_encoded, y_pred))
