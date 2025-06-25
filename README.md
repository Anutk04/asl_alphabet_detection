ASL Alphabet Detection using MediaPipe
This project performs **real-time American Sign Language (ASL) alphabet detection** using a webcam, 
powered by **MediaPipe** for hand landmark detection and a custom **machine learning classifier** for letter recognition. 
Detected letters are converted to live text with support for **space** and **delete** gestures.

Project Motivation & Importance:
Millions of people worldwide rely on **sign language** to communicate, but most of society doesn't understand it.  
This project aims to:
- Bridge the gap** between hearing and non-hearing individuals
- Allow seamless **human-computer interaction** using hand gestures
- Demonstrate how AI and CV can help build **assistive technology**
- Serve as a base for **future ASL-to-voice apps** or **gesture-controlled systems**


FEATURES:
✅ Real-time ASL alphabet recognition  
✅ Clean UI with text output and ROI box  
✅ Special gestures: `space` and `del`  
✅ Noise-reduction using frame stability logic  
✅ Model trained on hand landmarks (no raw images)  
✅ Written in Python using OpenCV, MediaPipe, and scikit-learn

How It Works:
1. Captures your hand via webcam inside a **Region of Interest (ROI)**.
2. Uses **MediaPipe** to extract 21 3D hand landmarks.
3. Feeds the landmark vector into a trained **ML classifier**.
4. Stabilizes predictions across frames to avoid flickering.
5. Outputs the predicted letter as part of a growing text string.
   

Tech Stack:
- **Python**
- **MediaPipe** – for real-time hand landmark detection
- **OpenCV** – for webcam and UI
- **scikit-learn** – to train and use the classifier
- **NumPy** – data handling

 Model Performance:
|---------------|----------------  ---------------- ----------- |
| Metric        | Value                                         |
|---------------|----------------- ----------------- -----------|
| Test Accuracy | ✅ ~78.5%                                    |
| Classes       | A–Z, space, del                               |
| Input         | 21 hand landmarks × 3 (x, y, z) = 63 features | 
| Model         | MLP Classifier (scikit-learn)                 |
|---------------|----------------  ---------------- ----------- |
