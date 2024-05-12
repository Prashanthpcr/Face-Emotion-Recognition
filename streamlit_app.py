import cv2
import numpy as np
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the trained model
model = load_model("modelk1.h5")
model.load_weights("modelk1.weights.h5")

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
                label_position = (x, y - 10)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    st.set_page_config(page_title="Real Time Face Emotion Detection", page_icon=":smiley:")

    st.title("Real Time Face Emotion Detection")
    st.sidebar.title("Navigation")

    # Render navigation menu
    page = st.sidebar.radio("Go to", ("Home", "Live Face Emotion Detection", "About"))

    if page == "Home":
        st.write("""
        * Welcome to the Real Time Face Emotion Detection app!
        * This app detects facial expressions in real time using your webcam.
        * To start, click on "Live Face Emotion Detection" in the sidebar.
        """)
    
    elif page == "Live Face Emotion Detection":
        st.subheader("Webcam Live Feed")
        st.write("""
        * Get ready with all the emotions you can express!
        * Click on the button below to start the live face emotion detection.
        """)
        webrtc_streamer(key="emotion-detection", video_processor_factory=VideoTransformer)

    elif page == "About":
        st.subheader("About")
        st.write("""
        * This app predicts facial emotions using a convolutional neural network (CNN).
        * The CNN model was trained using the Keras and TensorFlow libraries.
        * Face detection is achieved through OpenCV.
        """)

if __name__ == "__main__":
    main()
