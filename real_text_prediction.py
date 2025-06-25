import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Load trained model and label encoder
with open('asl_mediapipe_model.pkl', 'rb') as f:
    model, le = pickle.load(f)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)

# ROI settings
roi_start = (200, 100)   # (x1, y1)
roi_end = (440, 340)     # (x2, y2)

# State tracking
current_text = ""
last_letter = ""
stable_count = 0
THRESHOLD = 10
last_added_time = 0
min_time_between_letters = 1.5
CONFIDENCE_THRESHOLD = 0.75
DEL_CONFIDENCE_THRESHOLD = 0.90
DEL_THRESHOLD_EXTRA = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # Draw ROI
    cv2.rectangle(display_frame, roi_start, roi_end, (0, 255, 0), 2)
    roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    result = hands.process(roi_rgb)
    current_time = time.time()
    predicted_letter = ""
    prediction_confidence = 0

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Convert to full image coordinates
            landmarks = []
            for lm in handLms.landmark:
                x = lm.x * (roi_end[0] - roi_start[0])
                y = lm.y * (roi_end[1] - roi_start[1])
                z = lm.z
                landmarks.extend([x, y, z])

            if len(landmarks) == 63:
                probs = model.predict_proba([landmarks])[0]
                max_prob_index = np.argmax(probs)
                max_prob = probs[max_prob_index]
                letter = le.inverse_transform([max_prob_index])[0]

                predicted_letter = letter
                prediction_confidence = max_prob

                if max_prob >= CONFIDENCE_THRESHOLD:
                    if letter == last_letter:
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_letter = letter

                    if stable_count > THRESHOLD and (current_time - last_added_time > min_time_between_letters):
                        if letter.lower() == "del" and max_prob >= DEL_CONFIDENCE_THRESHOLD and stable_count > (THRESHOLD + DEL_THRESHOLD_EXTRA):
                            if current_text:
                                current_text = current_text[:-1]
                                print("✂️ Deleted last letter")
                        elif letter.lower() == "space":
                            current_text += " "
                            print("␣ Added space")
                        elif letter.lower() not in ["del", "space"]:
                            current_text += letter
                            print(f"➕ Added: {letter}")
                        last_added_time = current_time
                        stable_count = 0

            # Draw landmarks on ROI
            draw.draw_landmarks(roi, handLms, mp_hands.HAND_CONNECTIONS)

    # Paste processed ROI back to display frame
    display_frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]] = roi

    # Text area: semi-transparent box
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 400), (640, 480), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

    # Transcribed text
    cv2.putText(display_frame, current_text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (255, 255, 255), 3, cv2.LINE_AA)

    # Prediction box
    if predicted_letter:
        text = f"{predicted_letter} ({prediction_confidence:.2f})"
        cv2.putText(display_frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

    # UI instructions
    cv2.putText(display_frame, "ROI -> Put hand in green box", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    cv2.putText(display_frame, "Press 'r' to reset | 'q' to quit", (10, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)

    cv2.imshow("ASL to Text (ROI Enhanced)", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        current_text = ""
        print("🔄 Text cleared.")

cap.release()
cv2.destroyAllWindows()
