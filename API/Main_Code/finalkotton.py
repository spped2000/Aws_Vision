import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import json
from flask import Flask, jsonify, request

app = Flask(__name__)

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile


detect_fn = tf.saved_model.load('/home/tfs-people-analytics/Documents/kotton/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/')
PATH_TO_LABELS = '/home/tfs-people-analytics/Documents/Diamond_workspace/models/research/object_detection/data/person.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load the model
model_classification = load_model("new_model/converted_keras-2/keras_model.h5", compile=False)

#(2) Classification model prepare after passing detection model
def Classification(model_classify,model_detect,image_path):
    roi = show_inference(model_detect,image_path)
    # preprocess
    class_names = ["Cool", "Cute", "Sexy"]
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.fromarray(np.uint8(roi))
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model_classify.predict(data)[0]
    sorted_indexes = np.argsort(-prediction)  # Sort in descending order
    results = []
    for i in sorted_indexes:
        class_name = class_names[i].strip()
        confidence_score = float(prediction[i])
        result = {"class_name": class_name, "confidence_score": confidence_score}
        results.append(result)

    # Convert the list to JSON format
    json_str = json.dumps(results, indent=4)
    return json_str


# (3) Preprocess image before classification
def Resized_Ratio(image_detect):
    # Load the image
    img = Image.fromarray(image_detect)
    img.thumbnail((224, 224), Image.ANTIALIAS)
    new_img = Image.new("RGB", (224, 224), "Black")

    # Calculate the position to paste the resized image onto the new blank image
    x_offset = (224 - img.size[0]) // 2
    y_offset = (224 - img.size[1]) // 2

    # Paste the resized image onto the new blank image
    new_img.paste(img, (x_offset, y_offset))

    return new_img



#(4) Detection model
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  return output_dict



def show_inference(model, image_path):
    img = Image.open(image_path)
    img_jpg = img.convert('RGB')
    image_np = np.array(img_jpg)
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    person_detections = [i for i in range(len(output_dict['detection_classes'])) if output_dict['detection_classes'][i] == 1]

    boxes = output_dict['detection_boxes'][person_detections]
    im_height,im_width,_ = image_np.shape
    ymin, xmin, ymax, xmax = boxes[0]

    ymin = ymin * im_height
    xmin = xmin * im_width
    ymax = ymax * im_height
    xmax = xmax * im_width

    x = xmin
    y = ymin
    w = xmax - xmin
    h = ymax - ymin
    roi = image_np[int(y):int(y+h), int(x):int(x+w)]
    roi4classify = Resized_Ratio(roi)
    return roi4classify

def img_classification(image_path):
    # Start
    img_classify = Classification(model_classification,detect_fn,image_path)
    return img_classify


@app.route('/classify_image', methods=['POST'])
def classify_image():
    # Get the image from the request
    image = request.files.get('image')

    # Save the image to a temporary file
    image_path = 'tmp/image.jpg'
    image.save(image_path)

    # Call the image classification function
    img_classify = img_classification(image_path)
    res = json.loads(img_classify)
    # Return the classification results as JSON
    return jsonify(res)

if __name__ == '__main__':
    app.run()



   