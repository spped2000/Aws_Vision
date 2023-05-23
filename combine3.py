from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import cv2
import os
import numpy as np
import pytesseract

app = Flask(__name__)
api = Api(app)

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

#SWITCH PART
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

# LED Part
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

# OCR Part
def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

    # Apply a slight Gaussian blur to the image to reduce noise
    blurred_image = cv2.GaussianBlur(bilateral_image, (1, 1), 0)

    thresholded_image = cv2.threshold(blurred_image, 180, 255, cv2.THRESH_BINARY)[1]

    # Perform morphological operations to enhance the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)

    # Resize the image to make the text larger (optional, depending on the image)
    resized_image = cv2.resize(dilated_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return resized_image

def ocr_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Perform OCR on the preprocessed image
    ocr_result = pytesseract.image_to_string(preprocessed_image, lang='letsgodigital', config='--psm 6 -c tessedit_char_whitelist=0123456789')

    return ocr_result.strip()

# GAUGE
def measure_gauge_value(image):
    gauge_max = float(80)
    angle_max = float(360)
    gauge_units = "C"

    def avg_circles(circles, b):
        avg_x = 0
        avg_y = 0
        avg_r = 0
        for i in range(b):
            avg_x = avg_x + circles[0][i][0]
            avg_y = avg_y + circles[0][i][1]
            avg_r = avg_r + circles[0][i][2]

        avg_x = int(avg_x / (b))
        avg_y = int(avg_y / (b))
        avg_r = int(avg_r / (b))
        return avg_x, avg_y, avg_r

    def dist_2_pts(x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    threshold_img = 120  # 175
    threshold_ln = 150
    minLineLength = 30  # 50 previously
    maxLineGap = 1  # 8 previously

    # Distance from center coefficients
    diff1LowerBound = 0.15
    diff1UpperBound = 0.25

    # Distance from circle coefficients
    diff2LowerBound = 0.25
    diff2UpperBound = 2.0

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20)

    if circles is not None:
        a, b, c = circles.shape
        x, y, r = avg_circles(circles, b)

        gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        maxValue = 255

        # Threshold image to take better measurements
        th, dst2 = cv2.threshold(gray3, threshold_img, maxValue, cv2.THRESH_BINARY_INV)

        in_loop = 0
        lines = cv2.HoughLinesP(image=dst2, rho=1, theta=np.pi / 180, threshold=threshold_ln, minLineLength=minLineLength, maxLineGap=maxLineGap)
        final_line_list = []

        for i in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
                diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle

                if (diff1 > diff2):
                    temp = diff1
                    diff1 = diff2
                    diff2 = temp

                # Check if line is in range of circle
                if (((diff1 < diff1UpperBound * r) and (diff1 > diff1LowerBound * r) and (diff2 < diff2UpperBound * r)) and (diff2 > diff2LowerBound * r)):
                    line_length = dist_2_pts(x1, y1, x2, y2)
                    final_line_list.append([x1, y1, x2, y2])
                    in_loop = 1

        if (in_loop == 1):
            x1 = final_line_list[0][0]
            y1 = final_line_list[0][1]
            x2 = final_line_list[0][2]
            y2 = final_line_list[0][3]
            dist_pt_0 = dist_2_pts(x, y, x1, y1)
            dist_pt_1 = dist_2_pts(x, y, x2, y2)
            if (dist_pt_0 > dist_pt_1):
                x_angle = x1 - x
                y_angle = y - y1
            else:
                x_angle = x2 - x
                y_angle = y - y2

            # Finding angle using the arc tan of y/x
            res = np.arctan(np.divide(float(y_angle), float(x_angle)))

            # Converting to degrees
            res = np.rad2deg(res)
            if x_angle > 0 and y_angle > 0:  # in quadrant I
                final_angle = 270 - res
            if x_angle < 0 and y_angle > 0:  # in quadrant II
                final_angle = 90 - res
            if x_angle < 0 and y_angle < 0:  # in quadrant III
                final_angle = 90 - res
            if x_angle > 0 and y_angle < 0:  # in quadrant IV
                final_angle = 270 - res
            if x_angle == 0 or y_angle == 0:
                final_angle = abs(res)

            if (final_angle != 0):
                final_value = final_angle * (gauge_max / angle_max)
                final_value = int(final_value)
                final_value = str(final_value)
                return final_value + " " + gauge_units
            else:
                return "Can't determine gauge value"
        else:
            return "Can't find the indicator"
    else:
        return "Can't detect the gauge"


class OCR(Resource):
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
                ocr_result = ocr_image(image_path)
                return jsonify({"ocr_result": ocr_result})
            else:
                return {"error": "File could not be saved."}, 500


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


class GAUGE(Resource):
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
                gauge_result = measure_gauge_value(image_path)
                return jsonify({"gauge_vaule": gauge_result})
            else:
                return {"error": "File could not be saved."}, 500


api.add_resource(SWITCH, '/switch')
api.add_resource(LED, '/led')
api.add_resource(OCR, '/ocr')
api.add_resource(GAUGE, '/gauge')


if __name__ == '__main__':
    app.run(debug=True)