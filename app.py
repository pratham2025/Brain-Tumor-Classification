import sys
import os

if os.name == 'nt':  
    os.system('chcp 65001')  
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)

model = load_model('BrainTumor20EpochsCategorical10.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

IMG_WIDTH, IMG_HEIGHT = 64, 64

def get_class_name(class_no):
    return "No Brain Tumor" if class_no == 0 else "Yes Brain Tumor"

def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img = np.array(img)
        img = img / 255.0
        input_img = np.expand_dims(img, axis=0)
        return input_img
    except Exception as e:
        print(f"Error processing image {img_path}: {e}")
        return None

def get_result(img_path):
    input_img = preprocess_image(img_path)
    if input_img is not None:
        result = model.predict(input_img)
        predicted_class = np.argmax(result, axis=1)
        return predicted_class
    else:
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        value = get_result(file_path)
        if value is not None:
            result = get_class_name(value[0])
            return render_template('result.html', result=result)
        else:
            return jsonify({"error": "Error processing image"}), 400
    return None

if __name__ == '__main__':
    app.run(debug=True)
