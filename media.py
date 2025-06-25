# extract_landmarks_train.py

import os
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

train_path = 'ASL_Alphabet_Dataset/asl_alphabet_train'  # Your train set folder with A-Z subfolders
data = []
labels = []

classes = os.listdir(train_path)
for label in classes:
    folder = os.path.join(train_path, label)
    if not os.path.isdir(folder): continue

    print(f"Processing {label}...")
    for img_file in os.listdir(folder)[:300]:  # limit per class for speed
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            row = []
            for pt in hand.landmark:
                row += [pt.x, pt.y, pt.z]
            if len(row) == 63:
                data.append(row)
                labels.append(label)

np.save('landmark_data.npy', np.array(data))
np.save('landmark_labels.npy', np.array(labels))
print("✅ Landmark data saved.")
