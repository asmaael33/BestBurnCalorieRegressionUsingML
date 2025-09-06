import streamlit as st
import cv2
import numpy as np
from PIL import Image

import os
import urllib.request

def download_if_missing(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {dest_path}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")

AGE_PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/deploy_age.prototxt"
AGE_MODEL_URL = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel"
GENDER_PROTO_URL = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/deploy_gender.prototxt"
GENDER_MODEL_URL = "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel"


def load_models():
    download_if_missing(AGE_PROTO_URL, "models/age_deploy.prototxt")
    download_if_missing(AGE_MODEL_URL, "models/age_net.caffemodel")
    download_if_missing(GENDER_PROTO_URL, "models/gender_deploy.prototxt")
    download_if_missing(GENDER_MODEL_URL, "models/gender_net.caffemodel")

    age_net = cv2.dnn.readNetFromCaffe("models/deploy_age.prototxt", "models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe("models/deploy_gender.prototxt", "models/gender_net.caffemodel")
    return age_net, gender_net



AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

@st.cache_resource
def load_models():
    #age_net = cv2.dnn.readNetFromCaffe('models/deploy_age.prototxt', 'models/age_net.caffemodel')
    #gender_net = cv2.dnn.readNetFromCaffe('models/deploy_gender.prototxt', 'models/gender_net.caffemodel')
    return age_net, gender_net

def predict(image, age_net, gender_net):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.426337, 87.768914, 114.895847), swapRB=False)

        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward()[0].argmax()]

        age_net.setInput(blob)
        age = AGE_BUCKETS[age_net.forward()[0].argmax()]

        label = f"{gender}, {age}"
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Streamlit UI
st.title("🧠 Age & Gender Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    age_net, gender_net = load_models()
    result = predict(image, age_net, gender_net)

    st.image(result, caption="Prediction Result", use_column_width=True)
