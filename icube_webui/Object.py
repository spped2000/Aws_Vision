import tensorflow as tf
# print(tf.__version__)
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
from IPython.display import display
import pathlib


from flask import Flask, render_template, request, session, Response
import pandas as pd
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
import csv
import cv2
import base64
import json
import pickle
from werkzeug.utils import secure_filename
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from efficientnet_pytorch import EfficientNet




# Accepted image for to upload for object detection model
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg'}
 
app = Flask(__name__, template_folder='templateFiles', static_folder='staticFiles')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'You Will Never Guess'


################################################
# Create EfficientNet model
class_names = ['plane', '  car', ' bird', '  cat', ' deer', '  dog', ' frog', 'horse', ' ship', 'truck']
model1 = EfficientNet.from_pretrained('efficientnet-b0')

# Modify the last fully connected layer
num_ftrs = model1._fc.in_features
model1._fc = torch.nn.Linear(num_ftrs, 10) # set output to 10 classes

# Load saved state dictionary
state_dict = torch.load('efficientnet_b0.pth', map_location=torch.device('cpu'))

# Remove 'module.' prefix if present (for multi-GPU training)
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k[7:]: v for k, v in state_dict.items()}

# Load state dictionary into model
model1.load_state_dict(state_dict)

# Set model to evaluation mode
model1.eval()

transform = transforms.ToTensor()


# Define prediction function
def predict(image_path):
    size = (32,32)
    # Load image
    image = Image.open(image_path).resize(size)

    # Preprocess image
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model1(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

    return predicted_class





@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
 
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('index_upload_and_display_image_page2.html')
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    print(f' original : {img_file_path}')
    return render_template('show_image.html', user_image = img_file_path)
 
@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path = uploaded_image_path
    predicted_class = predict(uploaded_image_path)
    predicted_class = class_names[predicted_class]
    return render_template('show_image2.html', user_image=output_image_path, predicted_class=predicted_class)
 
# flask clear browser cache (disable cache)
# Solve flask cache images issue
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = False)