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

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    horizontal_flip=True,
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2, 
    zoom_range=0.2,
    fill_mode='nearest')

test_generator = test_datagen.flow_from_directory(
    VALID_DIR, 
    subset="training", 
    shuffle=True, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

with open(DATASET_DIR + '/categories.json', 'r') as f:
    cat_to_name = json.load(f)
    classes = list(cat_to_name.values())

print (classes)

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

model.compile(
   optimizer=tf.keras.optimizers.Adam(lr=0.01), 
   loss='categorical_crossentropy',
   metrics=['accuracy'])

results = model.evaluate_generator(test_generator)
print('test loss, test acc:', results)