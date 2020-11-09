import tensorflow as tf
import keras
import numpy as np
import os
import matplotlib.pyplot as plt

URL = r'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, "dogs")
val_cats_dir = os.path.join(validation_dir, 'cats')
val_dogs_dir = os.path.join(validation_dir, 'dogs')
train_total = len(os.listdir(train_cats_dir)) + len(os.listdir(train_dogs_dir))
val_total = len(os.listdir(val_cats_dir)) + len(os.listdir(val_dogs_dir))
print(train_total, val_total)

train_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=45, zoom_range=0.5, width_shift_range=.15, height_shift_range=.15)
val_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_image_gen.flow_from_directory(batch_size=16, directory=train_dir, target_size=(150, 150), class_mode='binary')
val_data = val_image_gen.flow_from_directory(batch_size=16, directory=validation_dir, target_size=(150, 150), class_mode='binary')

model = keras.Sequential(
    [keras.layers.Conv2D(32, (3, 3), padding='same', activation="relu", input_shape=(150, 150, 3)),
     keras.layers.MaxPooling2D(),

     keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
     keras.layers.MaxPooling2D(),

     keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
     keras.layers.MaxPooling2D(),

     keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
     keras.layers.MaxPooling2D(),

     keras.layers.Flatten(),
     keras.layers.Dense(64, activation='relu'),
     keras.layers.Dropout(.5),
     keras.layers.Dense(1, activation='sigmoid'),
     ]
)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(train_data, epochs=50, steps_per_epoch=train_total//16, validation_data=val_data, validation_steps=val_total//16)
model.save_weights('image_weights.h5')
epochs = range(50)
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.plot(epochs, hist.history['loss'])
plt.plot(epochs, hist.history['val_loss'], 'ro')
plt.subplot(1,2,2)
plt.plot(epochs, hist.history['accuracy'])
plt.plot(epochs, hist.history['val_accuracy'], 'ro')
plt.show()