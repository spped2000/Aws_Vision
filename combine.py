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

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    # loop over the unique components
    led_state = "OFF"
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
            led_state = "ON"

    # Save the labeled image
    labeled_image_path = "labeled_image1.jpg"
    cv2.imwrite(labeled_image_path, thresh)

    print("Labeled image saved:", labeled_image_path)
    print("LED state:", led_state)

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

api.add_resource(LED, '/led')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0')
