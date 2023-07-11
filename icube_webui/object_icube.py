from flask import Flask, render_template, request, jsonify
# from flask_restful import Resource, Api
from PIL import Image
import io
import base64
import numpy as np
import pytesseract
import cv2

app = Flask(__name__)
# api = Api(app)

def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    bilateral_image = cv2.bilateralFilter(gray_image, 11, 17, 17)

    # Apply a slight Gaussian blur to the image to reduce noise
    blurred_image = cv2.GaussianBlur(bilateral_image, (1, 1), 0)

    thresholded_image = cv2.threshold(blurred_image, 90, 255, cv2.THRESH_BINARY)[1]
    # j = Image.fromarray(thresholded_image)
    # # display(j)
    # j.save(f"ocr.png")



    # Perform morphological operations to enhance the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    dilated_image = cv2.dilate(thresholded_image, kernel, iterations=1)

    
    return dilated_image 


def ocr_image(image):
    print("OCR Section")
    # Read the image
    # image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # plt.imshow(preprocess_image)

    # Perform OCR on the preprocessed image
    ocr_result = pytesseract.image_to_string(preprocessed_image, lang='letsgodigital', config='--psm 6 -c tessedit_char_whitelist=0123456789')

    return "ocr number:" + ocr_result.strip() 


def detect_switch_state(image,count=None):
    print("Switch Section")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Then convert RGB to HSV
    img_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Black HSV mask
    lower_black, upper_black = np.array([0, 0, 185]), np.array([180, 55, 255])
    mask = cv2.inRange(img_hsv, lower_black, upper_black)

    # Draw a vertical line in the middle
    height, width = mask.shape[:2]
    middle_line_x = width // 2
    cv2.line(mask, (middle_line_x, 0), (middle_line_x, height), (255, 255, 255), 2)

    # Count white pixels on each side of the line
    left_side_pixels = np.count_nonzero(mask[:, :middle_line_x])
    right_side_pixels = np.count_nonzero(mask[:, middle_line_x:])

    # plt.imshow(mask, cmap='gray')
    print(f"left: {left_side_pixels}")
    print(f"right: {right_side_pixels}")
    j = Image.fromarray(mask)
    j.save(f"mask_{count}.png")

    # Determine the switch state
    if right_side_pixels > left_side_pixels:
        return f"Switch{count} ON"
    else:
        return f"Switch{count} OFF"
    

def is_led_on(image,count=None):
    print("LED Section")
    # Load the image
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Extract the region of interest
    roi = gray


    # Calculate the average pixel intensity within the ROI
    average_intensity = roi.mean()
    print (average_intensity)

    # Define a threshold value to determine the state of the LED
    threshold = 170  # Adjust this value as needed

    # Check if the average intensity is above the threshold
    if average_intensity > threshold:
        led_state = f"LED{count} ON"
    else:
        led_state = f"LED{count} OFF"
    
    return led_state


gauge_min = 0
gauge_min = float(gauge_min)
gauge_max = 80
gauge_max = float(gauge_max)
angle_min = 45
angle_min = float(angle_min)
angle_max = 350
angle_max = float(angle_max)
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


def take_measure(image,threshold_img, threshold_ln, minLineLength, maxLineGap, diff1LowerBound, diff1UpperBound, diff2LowerBound, diff2UpperBound):
    print("GAUGE Section")

    img = image
    frame = img
    scale_percent = 100  # percent of original size
    height = int(frame.shape[0] * scale_percent / 100)
    width = int(frame.shape[1] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20)

    if circles is not None:
        a, b, c = circles.shape
        x, y, r = avg_circles(circles, b)

        # Draw center and circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (x, y), 2, (0, 255, 0), 2, cv2.LINE_AA)

        separation = 10.0  # in degrees
        interval = int(360 / separation)
        p1 = np.zeros((interval, 2))  # set empty arrays
        p2 = np.zeros((interval, 2))

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
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 2 changed to 1
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

            # cv2.putText(img, "Indicator OK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            if (final_angle != 0):
                final_value = final_angle * (gauge_max / angle_max)
                final_value = int(final_value)
                final_value = str(final_value)
                return "Gauge Value: " + final_value + " " + gauge_units

        else:
            return "Can't find the indicator!"


    else:
        return "Can't see the gauge!"



threshold_img = 175  # 175
threshold_ln = 80
minLineLength = 10  # 50 previously
maxLineGap = 5  # 8 previously

# Distance from center coefficients
diff1LowerBound = 0.15
diff1UpperBound = 0.25

# Distance from circle coefficients
diff2LowerBound = 0.25
diff2UpperBound = 2.0



################### Process Section ########################

def get_Center(x, y, w, h): 
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


def jsonify_formatted_results(results):
    formatted_results = []
    for i, result in enumerate(results):
        result_dict = {
            "result": result
        }
        formatted_results.append(result_dict)
    return jsonify(formatted_results)


###################### Backend #################################


@app.route('/')
def home():
    return render_template('testdraw_roi.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    print(data)
    imgdata = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(imgdata))
    
    rects = data['rects']
    results = []
    for idx, rect in enumerate(rects):
        types = rect['type']
        startX, startY, w, h = rect['startX'], rect['startY'], rect['w'], rect['h']
        cropped_img = image.crop((startX, startY, startX+w, startY+h))
        # cropped_img.save(f"{types}_{idx}.jpg")
        crop = np.array(cropped_img)
        print(type(crop))
        if types == "SWITCH":
            SWITCH = detect_switch_state(crop)
            results.append(SWITCH)
            print(SWITCH)
        elif types == "LED":
            LED = is_led_on(cropped_img)
            results.append(LED)    
            print(LED)
        elif types == "OCR":
            OCR = ocr_image(cropped_img)
            results.append(OCR)  
            print(OCR)
        elif types == "GAUGE":
            GAUGE = take_measure(cropped_img,threshold_img, threshold_ln, minLineLength, maxLineGap, diff1LowerBound, diff1UpperBound, diff2LowerBound, diff2UpperBound)
            results.append(GAUGE)   
            print(GAUGE)  
        else:
            return {"error": "None."}, 500



    
    return jsonify_formatted_results(results), 200

if __name__ == '__main__':
    app.run(debug=True)
