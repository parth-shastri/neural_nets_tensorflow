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

#print(tfds.list_builders())

input_dir = 'seg_data/images'
target_dir = 'seg_data/annotations/trimaps'
img_size = (160, 160)
num_classes = 4
batch_size = 32
num_classes = 4


input_img_paths = sorted([os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith('.jpg')])
target_img_paths = sorted([os.path.join(target_dir, name) for name in os.listdir(target_dir) if name.endswith('.png')
                           and not name.startswith('._')])
#print(input_img_paths[:5], target_img_paths[:5])

x = Image.open(input_img_paths[9])
image = ImageOps.autocontrast(Image.open(target_img_paths[9]))


plt.subplot(2,2,1)
plt.imshow(x)
plt.subplot(2,2,2)
plt.imshow(image)
plt.show()

class Oxfordpets(keras.utils.Sequence):
    def __init__(self, input_img_paths, target_img_paths, batch_size, img_size):
        self.x = input_img_paths
        self.y = target_img_paths
        self.batch_size = batch_size
        self.img_size = img_size

    def __len__(self):
        return len(self.x)//self.batch_size

    def __getitem__(self, idx):
        i = idx*self.batch_size
        batch_input_img_paths = self.x[i:i+self.batch_size]
        batch_target_img_paths = self.y[i:i+self.batch_size]
        x = np.zeros((self.batch_size,)+self.img_size+(3,), dtype="float32")
        for i,path in enumerate(batch_input_img_paths):
            img = keras.preprocessing.image.load_img(path, target_size=self.img_size)
            x[i] = img
        y = np.zeros((self.batch_size,)+self.img_size+(1,), dtype="uint8")
        for i, path in enumerate(batch_target_img_paths):
            img = keras.preprocessing.image.load_img(path, target_size=self.img_size, color_mode='grayscale')
            y[i] = np.expand_dims(img, 2)
        return x,y

def call_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size+(3,))
    x = keras.layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = keras.layers.Conv2D(filters, 1, strides=2, padding='same')(previous_block_activation)
        x = keras.layers.add([x,residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding='same')(residual)
        x = keras.layers.add([x, residual])
        previous_block_activation = x

    outputs = keras.layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

    model = keras.Model(inputs, outputs)
    return model

keras.backend.clear_session()

model = call_model(img_size, num_classes)
model.summary()

validation = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-validation]
train_target_img_paths = target_img_paths[:-validation]
val_input_img_paths = input_img_paths[-validation:]
val_target_img_paths = target_img_paths[-validation:]

train_gen = Oxfordpets(train_input_img_paths, train_target_img_paths, batch_size, img_size)
val_gen = Oxfordpets(val_input_img_paths, val_target_img_paths, batch_size, img_size)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
epochs=15
callbacks = [keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
]
#model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
#model.save('unet-Xceptionstyle.h5')

val_gen = Oxfordpets(val_input_img_paths, val_target_img_paths, batch_size, img_size)

model = keras.models.load_model('saved_models/unet-Xceptionstyle.h5')

predictions = model.predict(val_gen)

def display(i):
    mask = np.argmax(predictions[i], axis =-1)
    mask = np.expand_dims(mask, axis = -1)
    img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    img = np.asarray(img)
    plt.imshow(img)
    plt.show()

i = 10
display(i)






        







