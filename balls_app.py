from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing.image import load_img
from keras.applications.mobilenet import preprocess_input
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt
from numpy import loadtxt
from keras.models import load_model
model = tf.keras.models.load_model('model.hdf5')
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('firstpage.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image = request.files['file'].read()
        img = Image.open(io.BytesIO(image))
        img = img.resize((150, 150))  
        img_array = np.array(img) / 255.0 
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        classes =  ['american_football', 'baseball', 'basketball', 'billiard_ball', 'bowling_ball', 'cricket_ball', 'football', 'golf_ball', 'hockey_ball', 'hockey_puck', 'rugby_ball', 'shuttlecock', 'table_tennis_ball', 'tennis_ball', 'volleyball']# Replace with your CNN model's class labels
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_class_index]
        return render_template('secondpage.html', prediction='{}'.format(predicted_class,plt.imshow(img)))

if __name__ == '__main__':
    app.run(debug=True)