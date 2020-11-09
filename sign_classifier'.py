import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import os


URL = r'https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/master.zip'
path_to_zip = tf.keras.utils.get_file('master.zip', extract=True, origin=URL)
PATH = os.path.join(os.path.dirname(path_to_zip), 'Dataset')
train = os.path.join(PATH, 'train')
test = os.path.join(PATH, 'test')
val = os.path.join(PATH, 'validation')
#print(os.listdir(train), os.listdir(test), os.listdir(val))

generator = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)
train_data= generator.flow_from_directory(train, target_size=(224, 224), batch_size=10)
validation_data = generator.flow_from_directory(val, target_size=(224, 224), batch_size=10)
test_data = generator.flow_from_directory(test, target_size=(224, 224), batch_size=10, shuffle=False)

mobile = keras.applications.mobilenet.MobileNet()
#mobile.summary()

last_layer = mobile.layers[-6].output
predictions = keras.layers.Dense(10, activation="softmax", name = 'predictions')(last_layer)
model = keras.Model(mobile.input, predictions)
model.summary()

for layers in model.layers[:-36]:
    layers.trainable = False

model.compile(optimizer=keras.optimizers.Adam(lr=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=150, steps_per_epoch=18, validation_data=validation_data, validation_steps=3)