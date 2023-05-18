from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import cv2
import os

app = Flask(__name__)
api = Api(app)


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
    threshold = 100  # Adjust this value as needed

    # Check if the average intensity is above the threshold
    if average_intensity < threshold:
        led_state = "ON"
    else:
        led_state = "OFF"
    
    return led_state

# led_state = is_led_on("/home/nont/icube/on3.jpg")
# print (led_state)

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

if __name__ == '__main__':
    app.run(debug=True)