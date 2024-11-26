import sys
import os
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')


model = load_model('BrainTumor20EpochsCategorical10.h5')


image = cv2.imread(r'D:\PROJECTS\btc\BrainTumor Classification DL\datasets\yes\y0.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img = Image.fromarray(image)

img = img.resize((64, 64))

img = np.array(img)

img = img / 255.0

img = np.expand_dims(img, axis=0)

predictions = model.predict(img)

predicted_class = np.argmax(predictions)

print(f"Predicted class: {predicted_class}")
