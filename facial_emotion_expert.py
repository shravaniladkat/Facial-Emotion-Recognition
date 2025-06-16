import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model without compiling it
model = load_model("fer2013_mini_XCEPTION.119-0.65.hdf5", compile=False)

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad','Surprise','Neutral' ]

# Load OpenCV's face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_emotion(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected")
        return

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Predict emotion
        prediction = model.predict(face)[0]
        emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the script with a test image
if __name__ == "__main__":
    detect_emotion("test_face.jpg")
    detect_emotion("test_face(1).jpg")
    detect_emotion("test_face(2).jpg")
    detect_emotion("test_face(3).jpg")
    detect_emotion("test_face(4).jpg")
    detect_emotion("test_face(5).jpg")
    detect_emotion("test_face.png")
