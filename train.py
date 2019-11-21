import tensorflow as tf
import json
import datetime
import os

DATASET_DIR = "Dataset/PlantVillage/"
TRAIN_DIR   = DATASET_DIR + "/train"
VALID_DIR   = DATASET_DIR + "/validation"
MODEL_DIR   = "./models/"
LOG_DIR     = "./logs/"
IMAGE_SIZE  = (299, 299)
BATCH_SIZE  = 16
LEARNING_RATE = 0.001

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

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    horizontal_flip=True,
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2, 
    zoom_range=0.2,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, 
    subset="training", 
    shuffle=True, 
    seed=42,
    color_mode="rgb", 
    class_mode="categorical",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

print("*"*50)
print()
print("*"*50)

model = tf.keras.Sequential([
    tf.keras.applications.InceptionV3(input_shape=IMAGE_SIZE+(3,),  include_top=False, weights='imagenet'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])

latest = tf.train.latest_checkpoint(MODEL_DIR)
if latest:
    print("*"*50)
    print("Load weight from last")
    print(latest)
    model.load_weights(latest)
    print("*"*50)

model.compile(
   optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), 
   loss='categorical_crossentropy',
   metrics=['accuracy'])

model.summary()

print("*"*50)
print()
print("*"*50)

now = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

log_dir = LOG_DIR + now
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='batch')

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

checkpoint_path = MODEL_DIR + "/cp-" + now + "-{epoch:04d}.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    save_freq='epoch')

EPOCHS=10 #@param {type:"integer"}
STEPS_EPOCHS = train_generator.samples//train_generator.batch_size

history = model.fit_generator( 
          train_generator,
          steps_per_epoch=STEPS_EPOCHS,
          epochs=EPOCHS,
          callbacks=[tensorboard_callback, checkpoint_callback])