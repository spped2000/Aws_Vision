from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import cv2
import os
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils

app = Flask(__name__)
api = Api(app)

def detect_switch_state(image_path):
    image = cv2.imread(image_path)
    # image = cv2.flip(image, 1) #OFF

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV space

    # Black HSV mask
    lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 65])
    mask = cv2.inRange(img_hsv, lower_black, upper_black)

    # Draw a vertical line in the middle
    height, width = mask.shape[:2]
    middle_line_x = width // 2
    cv2.line(mask, (middle_line_x, 0), (middle_line_x, height), (255, 255, 255), 2)

    # Count white pixels on each side of the line
    left_side_pixels = np.count_nonzero(mask[:, :middle_line_x])
    right_side_pixels = np.count_nonzero(mask[:, middle_line_x:])

    # Determine the switch state
    if right_side_pixels > left_side_pixels:
        return "ON"
    else:
        return "OFF"


def is_led_on(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract the region of interest
    roi = gray
    # roi = cv2.resize(roi, (0, 0), fx = 0.1, fy = 0.1)

    # cv2.imshow("gray", roi)
    # cv2.waitKey(0)

    # Calculate the average pixel intensity within the ROI
    average_intensity = roi.mean()
    print (average_intensity)

    # Define a threshold value to determine the state of the LED
    threshold = 150  # Adjust this value as needed

    # Check if the average intensity is above the threshold
    if average_intensity > threshold:
        led_state = "ON"
    else:
        led_state = "OFF"
    
    return led_state

class LED(Resource):
    def post(self):
        if 'file' not in request.files:
            return {"error": "No file part in the request."}, 400
        file = request.files['file']
        if file.filename == '':
            return {"error": "No file selected for uploading"}, 400
        else:
            upload_dir = "uploads"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            image_path = os.path.join(upload_dir, file.filename)
            file.save(image_path)
            if os.path.exists(image_path):
                led_result = is_led_on(image_path)
                return jsonify({"led_result": led_result})
            else:
                return {"error": "File could not be saved."}, 500



class SWITCH(Resource):
    def post(self):
        if 'file' not in request.files:
            return {"error": "No file part in the request."}, 400
        file = request.files['file']
        if file.filename == '':
            return {"error": "No file selected for uploading"}, 400
        else:
            upload_dir = "uploads"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            image_path = os.path.join(upload_dir, file.filename)
            file.save(image_path)
            if os.path.exists(image_path):
                switch_result = detect_switch_state(image_path)
                return jsonify({"switch_result": switch_result})
            else:
                return {"error": "File could not be saved."}, 500

api.add_resource(SWITCH, '/switch')
api.add_resource(LED, '/led')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
