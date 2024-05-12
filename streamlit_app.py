import cv2
import numpy as np
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array

def main():
    st.title("Emotion Detection")

    model = load_model("modelk1.keras")
    model.load_weights('modelk1.weights.keras')
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Unable to open webcam.")
        return

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Error: Unable to capture frame.")
            break

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
                emotion_detection = ('Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral')
                emotion_prediction = emotion_detection[max_index]
                cv2.putText(res, "Sentiment: {}".format(emotion_prediction), (0, height//6 + 22 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 255), 2)
                confidence = np.round(np.max(predictions[0]) * 100, 1)
                cv2.putText(res, "Confidence: {}%".format(confidence), (width - 200, height//6 + 22 + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 255), 2)
        except:
            pass

        frame[0:int(height/6), 0:int(width)] = res

        stframe.image(frame, channels="BGR", use_column_width=True)

        if st.button("Stop"):
            break

    cap.release()

if __name__ == "__main__":
    main()
