😀 Real-time Emotion Detection with Emoji Mirror

This project detects facial emotions in real-time using a webcam feed. It leverages Mediapipe for face landmark detection, a pretrained deep learning model (Mini Xception trained on FER-2013) for emotion classification, and overlays emojis corresponding to emotions on the live video.

---

🚀 Features

• Real-time face detection using Mediapipe FaceMesh
• Emotion classification with pretrained Mini Xception model
• Emoji overlay for emotions like happy, sad, angry, surprise, fear, disgust, neutral
• Live webcam feed with bounding boxes and labels
• Smooth integration of OpenCV and PIL for drawing emojis

---

🛠 Tech Stack

• Python 
• OpenCV (cv2)
• Mediapipe
• TensorFlow / Keras
• Pillow (PIL) for emoji overlay
• NumPy

---

🔎 Workflow

1. Capture webcam frames using OpenCV
2. Detect face landmarks with Mediapipe FaceMesh
3. Extract face region of interest (ROI)
4. Preprocess ROI (grayscale, resize, normalize)
5. Predict emotion using pretrained Mini Xception model
6. Map detected emotion to emoji
7. Overlay emoji and emotion text on video feed
8. Display final output until user quits with q

---

🎯 Use Cases

* Fun emoji mirror applications
* Virtual classrooms and interactive learning
* Human-computer interaction
* Mental health monitoring systems
* Emotion-aware chatbots and gaming
