import tkinter as tk
# from tkinter import filedialog, Label, Button, Entry, W, END, Canvas
from tkinter import *
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import time

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

gui = tk.Tk()
gui.resizable(False, False)
gui.title("Disease Identification")

Label(gui, text="Image path: ").grid(row=0, sticky=W)
txtbPath = Entry(gui)
txtbPath.grid(row=0, column=1, columnspan=2, sticky=W)

Label(gui, text="True label: ").grid(row=1, sticky=W)
txtbTrueLabel = Entry(gui)
txtbTrueLabel.grid(row=1, column=1, columnspan=2, sticky=W)

Label(gui, text="Predited label: ").grid(row=2, sticky=W)
txtbLabel = Entry(gui)
txtbLabel.grid(row=2, column=1,sticky=W)

Label(gui, text="Confident: ").grid(row=3, column=0, sticky=W)
txtbScore = Entry(gui)
txtbScore.grid(row=3, column=1, sticky=W)

Label(gui, text="Result: ").grid(row=4, column=0, sticky=W)
txtbCorrect = Entry(gui)
txtbCorrect.grid(row=4, column=1, sticky=W)

def OpenFile():
    path = filedialog.askopenfilename()
    print (path)
    txtbPath.delete(0,END)
    txtbPath.insert(0,path)

    cv_img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    cv2.resize(cv_img, (400, 400))
    photo = ImageTk.PhotoImage(image = Image.fromarray(cv_img))
    # item = canvas.create_image(0, 0, image=photo, anchor=NW)
    # canvas.tag_raise(item)
    lb = Label(gui, image = photo)
    lb.image = photo
    lb.grid(row=5, columnspan=3)

    true_label = path.split('/')[-2]
    print(true_label)

    txtbTrueLabel.delete(0,END)
    txtbTrueLabel.insert(0,true_label)

    img = load_image(path)
    
    probabilities = model.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)

    print("probabilities: ")
    print(probabilities[class_idx])
    txtbScore.delete(0,END)
    txtbScore.insert(0,str(probabilities[class_idx]))

    print("PREDICTED: class: %s, confidence: %f" % (classes[class_idx], probabilities[class_idx]))
    txtbLabel.delete(0,END)
    txtbLabel.insert(0,str(classes[class_idx]))

    txtbCorrect.delete(0,END)
    if(txtbLabel.get() == txtbTrueLabel.get()):
        txtbCorrect.insert(0,"Right Prediction")
    else:
        txtbCorrect.insert(0,"Wrong Prediction")

btnOpen = Button(gui, text=' Open ', fg='black', command=lambda: OpenFile(), height=1, width=7) 
btnOpen.grid(row=0, column=2)

gui.mainloop()