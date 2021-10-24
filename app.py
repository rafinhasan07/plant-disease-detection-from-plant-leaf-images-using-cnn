from __future__ import division, print_function

#Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf


# Keras

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask App

app = Flask(__name__)

# Model Path

MODEL_PATH ='plant_disease_CNN_model.h5'

# Loading the trained model
model = load_model(MODEL_PATH)

# Disease Classes
class_name = {0: 'Apple: Apple scab',
              1: 'Apple: Apple Black Rot',
              2: 'Apple: Cedar Apple Rust',
              3: 'Apple: Healthy',
              4: 'Blueberry: Healthy',
              5: 'Cherry (including_sour): Healthy',
              6: 'Cherry (including_sour): Powdery Mildew',
              7: 'Corn (maize): Cercospora leaf spot- Gray leaf spot',
              8: 'Corn (maize): Common rust',
              9: 'Corn (maize): Healthy',
              10: 'Corn (maize): Northern Leaf Blight',
              11: 'Grape: Black rot',
              12: 'Grape (Esca): Black Measles',
              13: 'Grape: Healthy',
              14: 'Grape: Leaf blight (Isariopsis Leaf Spot)',
              15: 'Orange: Haunglongbing (Citrus greening)',
              16: 'Peach: Bacterial spot',
              17: 'Peach: Healthy',
              18: 'Pepper bell: Bacterial_spot',
              19: 'Pepper bell: Healthy',
              20: 'Potato: Early blight',
              21: 'Potato: Healthy',
              22: 'Potato: Late blight',
              23: 'Raspberry: Healthy',
              24: 'Soybean: Healthy',
              25: 'Squash: Powdery mildew',
              26: 'Strawberry: Healthy',
              27: 'Strawberry: Leaf scorch',
              28: 'Tomato: Bacterial spot',
              29: 'Tomato: Early blight',
              30: 'Tomato: Healthy',
              31: 'Tomato: Late blight',
              32: 'Tomato: Leaf Mold',
              33: 'Tomato: Septoria leaf spot',
              34: 'Tomato: Spider mites- Two spotted spider mite',
              35: 'Tomato: Target Spot',
              36: 'Tomato: Tomato mosaic virus',
              37: 'Tomato: Tomato Yellow Leaf Curl Virus'}


# Prediction

def model_predict(img_path, model, class_name):
    img = image.load_img(img_path, target_size=(128, 128))
    img_arr = image.img_to_array(img)

    img_arr = img_arr / 255
    img_preprocessed = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_preprocessed)

    classes_identifier = np.argmax(prediction, axis=1)

    predected_class = class_name[classes_identifier[0]]

    return predected_class


# Routes

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    # About page
    return render_template('about.html')


# Image Upload and Processing

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model, class_name)
        result = preds
        return render_template('index.html', result=result)
    return None



if __name__ == '__main__':
    app.run()