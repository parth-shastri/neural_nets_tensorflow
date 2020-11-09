import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
traindir = r'\train'
valdir = r'\validation'
testdir = r'\test'
PATH = os.path.join(r'C:\Users\shast\.keras\datasets', 'cats_and_dogs_filtered')
#os.makedirs(r'C:\Users\shast\.keras\datasets\cats-und-dogs')
path = r'C:\Users\shast\.keras\datasets\cats-und-dogs'
#os.makedirs(path+traindir)
#os.makedirs(path+valdir)
#os.makedirs(path+testdir)
os.chdir(path)
train_dir = 'train'
test_dir = 'test'
val_dir = 'validation'

'''train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, "dogs")
val_cats_dir = os.path.join(validation_dir, 'cats')
val_dogs_dir = os.path.join(validation_dir, 'dogs')
train_total = len(os.listdir(train_cats_dir)) + len(os.listdir(train_dogs_dir))
val_total = len(os.listdir(val_cats_dir)) + len(os.listdir(val_dogs_dir))'''

image_height, image_width = 150, 150
batch_size = 10
epochs= 30

generator = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
train_data = generator.flow_from_directory(train_dir, target_size=(image_width, image_height), batch_size=batch_size)
val_data = generator.flow_from_directory(val_dir, target_size=(image_width, image_height), batch_size=batch_size)
test_data = generator.flow_from_directory(test_dir, target_size=(image_width, image_height), batch_size=batch_size, shuffle=False)

mobile = keras.applications.mobilenet.MobileNet()

last_layer = mobile.layers[-6].output
out = keras.layers.Dense(2, activation='softmax')(last_layer)
model = keras.Model(mobile.input, out)

for layer in model.layers[:-5]:
    layer.trainable = False
model.summary()
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
#hist = model.fit(train_data, epochs=epochs, steps_per_epoch=8, validation_data = val_data, validation_steps=4)
model.save('mobilenet_finetune.h5')

test_labels = test_data.classes
print(test_labels)
model = keras.models.load_model('mobilenet_finetune.h5')

predictions = model.predict(test_data, steps=2, verbose=0)
print(predictions)
