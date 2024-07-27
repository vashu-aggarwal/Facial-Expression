import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
import os

# Function to load the model
@st.cache_resource
def load_model():
    model_json_file = "facialemotionmodeltest.json"
    model_weights_file = "facialemotionmodeltest.h5"

    if os.path.exists(model_json_file) and os.path.exists(model_weights_file):
        with open(model_json_file, "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(model_weights_file)
        return model
    else:
        raise FileNotFoundError("Model files not found. Ensure 'facialemotionmodel.json' and 'facialemotionmodel.h5' are in the same directory.")

# Load the model once
model = load_model()

# Load the face cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Streamlit App
def home():
    st.title("Btech 4th Semester Mini project 2024")
    st.subheader("Facial Expression Detection")

    st.markdown("<h3 style='color: #0066cc;'>Welcome to the Home Page!</h3>", unsafe_allow_html=True)
    st.write("This is a simple app to detect facial expressions from the webcam feed.")
    st.write("To get started, navigate to the 'Predict' page using the navigation bar on the left.")
    st.write("Submitted by: Vashu Aggarwal")

def predict_page():
    st.title("Facial Expression Detection - Predict Page")
    st.markdown(
        f"<style>body {{background-image: linear-gradient(to right, #007BFF, #ADD8E6);}}</style>", unsafe_allow_html=True
    )
    st.markdown("<h3 style='color: #0066cc;'>Enable the webcam to detect facial expressions:</h3>", unsafe_allow_html=True)

    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")

    FRAME_WINDOW = st.image([])
    expression_placeholder = st.empty()

    if 'webcam' not in st.session_state:
        st.session_state.webcam = None

    if start_button and st.session_state.webcam is None:
        st.session_state.webcam = cv2.VideoCapture(0)

    if stop_button and st.session_state.webcam is not None:
        st.session_state.webcam.release()
        st.session_state.webcam = None
        FRAME_WINDOW.empty()
        expression_placeholder.empty()

    if st.session_state.webcam is not None and st.session_state.webcam.isOpened():
        run = True
    else:
        run = False

    while run:
        _, im = st.session_state.webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            expression_placeholder.markdown(
                f"<h3 style='color: #0066cc;'>Detected Expression: {prediction_label}</h3>", unsafe_allow_html=True
            )

        FRAME_WINDOW.image(im)

# Create a simple navigation bar
pages = {"Home": home, "Predict": predict_page}
navigation = st.sidebar.radio("Navigation", list(pages.keys()))

# Display the selected page
pages[navigation]()
