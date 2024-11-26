import sys
import os

if os.name == 'nt':
    os.system('chcp 65001')
    sys.stdout.reconfigure(encoding='utf-8')

import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input

IMAGE_DIRECTORY = 'D:/PROJECTS/btc/BrainTumor Classification DL/datasets'
IMG_WIDTH, IMG_HEIGHT = 64, 64
NUM_CLASSES = 2

def load_images(image_directory, label_value):
    images = []
    labels = []
    for image_name in os.listdir(image_directory):
        if image_name.endswith(('.jpg', '.jpeg', '.png')):
            try:
                image_path = os.path.join(image_directory, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    images.append(np.array(image))
                    labels.append(label_value)
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
    return np.array(images), np.array(labels)

no_tumor_images, no_tumor_labels = load_images(os.path.join(IMAGE_DIRECTORY, 'no'), 0)
yes_tumor_images, yes_tumor_labels = load_images(os.path.join(IMAGE_DIRECTORY, 'yes'), 1)

dataset = np.concatenate((no_tumor_images, yes_tumor_images), axis=0)
label = np.concatenate((no_tumor_labels, yes_tumor_labels), axis=0)

x_train, x_test, y_train, y_test = train_test_split(dataset / 255.0, label, test_size=0.2, random_state=0)

y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

model = Sequential([
    Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test), shuffle=True)

model.save('BrainTumor20EpochsCategorical10.h5')
