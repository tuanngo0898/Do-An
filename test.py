import tensorflow as tf
import json
import datetime
import os
import cv2
import numpy as np
import sys

DATASET_DIR = "Dataset/PlantVillage/"
TRAIN_DIR   = DATASET_DIR + "/train"
VALID_DIR   = DATASET_DIR + "/validation"
MODEL_DIR   = "./models/"
LOG_DIR     = "./logs/"
IMAGE_SIZE  = (299, 299)
BATCH_SIZE  = 16
FV_SIZE     = 2048

def load_image(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]) )
    img = img /255.0
    return img

print("*"*50)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

print("*"*50)
print()
print("*"*50)

with open(DATASET_DIR + '/categories.json', 'r') as f:
    cat_to_name = json.load(f)
    classes = list(cat_to_name.values())

print (classes)

print("*"*50)
print()
print("*"*50)

model = tf.keras.Sequential([
    tf.keras.applications.InceptionV3(input_shape=IMAGE_SIZE+(3,),  include_top=False),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

latest = tf.train.latest_checkpoint(MODEL_DIR)
if latest:
    print("Load weight from last")
    print(latest)
    model.load_weights(latest)
else:
    print("No weights. Exits")
    exit()

model.summary()

img = load_image(sys.argv[1])
probabilities = model.predict(np.asarray([img]))[0]
print("probabilities: ")
print(probabilities)
class_idx = np.argmax(probabilities)
print("PREDICTED: class: %s, confidence: %f" % (classes[class_idx], probabilities[class_idx]))