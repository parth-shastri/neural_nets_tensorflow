import tensorflow as tf
import keras
import numpy as np
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import os
from IPython.display import Image, display
from PIL import ImageOps
from PIL import Image
from keras.preprocessing import image
import matplotlib.pyplot as plt
import random
from keras import layers

input_dir = 'seg_data/images'
target_dir = 'seg_data/annotations/trimaps'
img_size = (160, 160)
num_classes = 4
batch_size = 8

input_img_paths = sorted([os.path.join(input_dir, path) for path in os.listdir(input_dir) if path.endswith('.jpg')])
target_img_paths = sorted([os.path.join(target_dir, path) for path in os.listdir(target_dir) if path.endswith('.png') and not path.startswith('._')])
print(input_img_paths[:5], target_img_paths[:5])

class Oxfordpets(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_path, target_path):
        self.x = input_path
        self.y = target_path
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.x)//self.batch_size

    def __getitem__(self, idx):
        i = idx*self.batch_size
        batch_input_paths = self.x[i:i + self.batch_size]
        batch_target_paths = self.y[i:i + self.batch_size]
        x = np.zeros((batch_size,)+self.img_size+(3,), dtype='float32')
        for i, path in enumerate(batch_input_paths):
            img = keras.preprocessing.image.load_img(path, target_size=self.img_size)
            x[i] = img
        y = np.zeros((self.batch_size,)+self.img_size+(1,), dtype='uint8')
        for i, path in enumerate(batch_target_paths):
            img = keras.preprocessing.image.load_img(path, target_size=self.img_size, color_mode='grayscale')
            y[i] = np.expand_dims(img, 2)
        return x, y

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size+(3,))
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    res1 = x
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    res2 = x
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)

    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    res3 = x
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)

    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(256, 2, padding='same', activation='relu')(x)
    x = layers.concatenate([res3, x], axis=3)
    x = layers.Conv2D(256, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(128, 2, padding='same', activation='relu')(x)
    x = layers.concatenate([res2, x], axis=3)
    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 2, padding='same', activation='relu')(x)
    x = layers.concatenate([res1, x], axis=3)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)




    model = keras.Model(inputs, outputs)
    return model
keras.backend.clear_session()

model = get_model(img_size, num_classes)
print(model.summary())

validation = 1600
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-validation]
train_target_img_paths = target_img_paths[:-validation]
val_input_img_paths = input_img_paths[-validation:1000]
val_target_img_paths = target_img_paths[-validation:1000]

train_gen = Oxfordpets(batch_size, img_size, train_input_img_paths, train_target_img_paths)
val_gen = Oxfordpets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
epochs=5
#callbacks = [keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)]
model.fit(train_gen, epochs=epochs, validation_data=val_gen)
#model.save('unet-original(selfmade).h5')
model.save('saved_models/new_unet(self).h5')
val_gen = Oxfordpets(batch_size, img_size, val_input_img_paths, val_target_img_paths)

#model = keras.models.load_model('saved_models/unet-original(selfmade).h5')

#predictions = model.predict(val_gen)


'''def display(i):
    mask = np.argmax(predictions[i], axis =-1)
    mask = np.expand_dims(mask, axis = -1)
    img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    img = np.asarray(img)
    plt.imshow(img)
    plt.show()

i = 10
display(i)'''

