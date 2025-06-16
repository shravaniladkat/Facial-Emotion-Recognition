# Facial Emotion Recognition 😄😢😡

This project detects human emotions from face images using a pre-trained CNN model (`mini_XCEPTION`) and OpenCV.

## 🔍 Features
- Face detection with OpenCV Haar Cascade
- Emotion prediction using FER-2013 trained model
- Recognizes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## 🛠 Requirements
Install dependencies with:
```bash
pip install opencv-python tensorflow numpy
````
## How to Use
Make sure you have:

The model file: fer2013_mini_XCEPTION.119-0.65.hdf5

Sample images: test_face.jpg, test_face(1).jpg, etc.

Run the script:
```bash
python emotion_detect.py
```
## Output
-Detects faces in images
-Annotates each face with the predicted emotion
-Displays the image with rectangles and labels
