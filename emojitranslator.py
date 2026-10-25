import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

# Load Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load pretrained emotion detection model
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Emotion labels and emoji map
emotion_els = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    : 'neutral',
}
emoji_dict = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'sad': '😢',
    'surprise': '😲',
    'neutral': '😐'
}

# Load emoji font
font = ImageFont.truetype("seguiemj.ttf", 40)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror flip the frame (horizontal flip)
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Calculate bounding box from landmarks
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size == 0:
                continue

            # Preprocess face for model
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(gray_face, (64, 64))
            normalized_face = resized_face / 255.0
            reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))

            # Predict emotions
            emotions_probabilities = model.predict(reshaped_face, verbose=0)
            emotion_index = np.argmax(emotions_probabilities)
            emotion = emotion_labels[emotion_index]
            confidence = emotions_probabilities[0][emotion_index] * 100  # %

            emoji = emoji_dict.get(emotion, '😐')

            # Draw emoji using PIL
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            emoji_x = x_min + (x_max - x_min) // 2 - 20
            emoji_y = y_min - 60
            draw.text((emoji_x, emoji_y), emoji, font=font, embedded_color=True)

            # Convert back to OpenCV
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Draw emotion label + confidence
            cv2.putText(frame, f"{emotion} ({confidence:.1f}%)",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

    cv2.imshow('Emoji Mirror', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()