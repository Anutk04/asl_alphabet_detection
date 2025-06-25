# train_model.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
X = np.load('landmark_data.npy')
y = np.load('landmark_labels.npy')

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train MLP
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"✅ Validation Accuracy: {acc:.4f}")

# Save model + label encoder
with open('asl_mediapipe_model.pkl', 'wb') as f:
    pickle.dump((clf, le), f)
print("✅ Model saved as 'asl_mediapipe_model.pkl'")
