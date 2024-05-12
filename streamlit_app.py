import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained model
model = load_model('modelk1.keras')
model.load_weights('modelk1.weights.keras.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

st.title("Emotion Detection")

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()
    
    height, width, _ = frame.shape
    sub_img = frame[0:int(height/6), 0:int(width)]

    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 0)
    
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image)
    
    try:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(res, f"Sentiment: {emotion_prediction}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 255), 2)
            confidence = f"Confidence: {np.round(np.max(predictions[0])*100,1)}%"
            cv2.putText(res, confidence, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 255), 2)
    except Exception as e:
        st.error(f"Error: {e}")
    
    frame[0:int(height/6), 0:int(width)] = res
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    st.image(frame, channels="RGB", use_column_width=True)
