import os
import cv2
import numpy as np
import joblib
import random
import requests
import string
import streamlit as st

from claptcha import Claptcha           # Claptcha for generating CAPTCHA images
from keras.models import load_model, Sequential
from PIL import Image
from skimage.feature import hog

mono_path = "FreeMono.ttf"              # Font filename from the URL
cnn_clf = load_model('cnn_model.hdf5')  # Load the HDF5 model
svm_clf = joblib.load('svm_model.pkl')  # Load the pkl model
image_margin = (0, 0)
char_width = 37.5
image_size = (int(4 * char_width), 90)
file_path = 'generated_image.png'

# Define dictionary for character mapping (dict_class)
dict_class = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
              12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M',
              23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
              34: 'Y', 35: 'Z'}

st.title("CNN vs SVM Captcha Classifier")

uploaded_image = st.file_uploader("Upload a captcha text image", type=["jpg", "jpeg", "png"])

col1, col2, col3, col4, col5 = st.columns(5)
with col3:
    st.write('Or generate one')

# Options for generating a CAPTCHA
col1, col2, col3 = st.columns(3)
with col1:
    custom_text = st.text_input("Custom Text")
with col2:
    captcha_length = st.number_input('Or use random text, random text len', min_value=1, value=4)
with col3:
    image_noise = st.number_input("Image Noise", min_value=0.0, value=0.2)

if custom_text:
    captcha_length = len(custom_text)


placeholder = st.empty()


def main(uploaded_image, captcha_length, custom_text):

    if st.button("Generate CAPTCHA"):
        uploaded_image = None
        if custom_text:
            image = generate_captcha(custom_text, image_noise)
        elif captcha_length:
            random_text = random_string(captcha_length)
            image = generate_captcha(random_text, image_noise)
        image = np.array(image)
        display_image(image, captcha_length=captcha_length, uploaded=False)


    elif uploaded_image is not None:
        image = np.array(Image.open(uploaded_image))
        captcha_length = int(image.shape[1] / char_width)
        display_image(image, captcha_length=captcha_length, uploaded=True)

def display_image(image, captcha_length, uploaded=False):

    # Create two columns to display the images side by side
    with placeholder.container():
        col1, col2 = st.columns(2)

        # Display the captcha image in the first column
        with col1:
            st.image(image, caption="CAPTHCA Image", use_column_width=True)
        # Add a button to make a prediction in the second column
        with col2:
            # Perform your image prediction here
            cnn_prediction, svm_prediction = predict(image, captcha_length)

            # Draw the predicted text on the image
            y = 4
            w = 32
            h = 81

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Draw bounding boxes around each letter
            for index, letter in enumerate(cnn_prediction):
                x = 37 * index + 2
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Put the prediction letters inside the boxes
                cv2.putText(image, f'{letter}', (int(x + (w / 2) - 4), y + 25), font, 0.5, (255, 0, 0), 2)
                cv2.putText(image, f'{svm_prediction[index]}', (int(x + (w / 2) - 4), y + h - 6), font, 0.5, (0, 0, 255), 2)

            cv2.putText(image, 'CNN predict', (y, y + 8), font, 0.5, (255, 0, 0), 1)
            cv2.putText(image, 'SVM predict', (y, y + h - 20), font, 0.5, (0, 0, 255), 1)

            # Display the image with the predicted text in the second column
            st.image(image, caption="Image with Prediction", use_column_width=True)

            # Display the predicted text
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"CNN Predict: ")
            with col2:
                st.code(''.join(cnn_prediction), language="python")

            col1, col2 = st.columns(2)
            with col1:
                st.text(f"SVM Predict: ")
            with col2:
                st.code(''.join(svm_prediction), language="python")

def predict(image, captcha_length):
    cnn_combine = []
    svm_combine = []

    for sliced_image in slice_image(process_captcha_image(image), captcha_length):  # Slicing the image
        feature_vector = hog(cv2.resize(sliced_image, (38,90)), pixels_per_cell=(8, 8))
        svm_prediction = dict_class[svm_clf.predict([feature_vector])[0]]
        svm_combine.append(svm_prediction)

        resized = cv2.resize(sliced_image, (32,32))
        img = np.expand_dims(resized, axis=2)
        class_probs = cnn_clf.predict(np.expand_dims(img, axis=0))
        cnn_prediction = dict_class[np.argmax(class_probs, axis=1)[0]]
        cnn_combine.append(cnn_prediction)

    return cnn_combine, svm_combine

def slice_image(image, captcha_length):
    sliced_image = []
    for i in range(captcha_length):
        # Extract and process individual characters from the CAPTCHA
        start_col = int(char_width * i)
        end_col = int(start_col + char_width)
        sliced_image.append(image[0:0 + image_size[1], start_col:end_col])
    return sliced_image

# Function to generate a CAPTCHA
def generate_captcha(text, image_noise):
    # Create a CAPTCHA image using the Claptcha library
    image_size = (int(char_width * len(text)), 90)
    c = Claptcha(text.upper(), mono_path, image_size, resample=Image.BICUBIC, noise=image_noise)
    c.margin = image_margin
    _, image = c.image
    return image

def random_string(length):
    """Helper function to generate random strings"""
    rndLetters = (random.sample(string.ascii_uppercase + string.digits, length))
    return "".join(rndLetters)

def process_captcha_image(image):
    """Function to preprocess individual letters of CAPTCHA images"""
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(gray_scale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    return image


if __name__ == "__main__":
    main(uploaded_image, captcha_length, custom_text)
