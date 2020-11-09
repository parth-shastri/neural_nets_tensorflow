import tensorflow as tf
import keras
import numpy as np
import os
import matplotlib.pyplot as plt
PATH = os.path.join(r'C:\Users\shast\.keras\datasets', 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, "dogs")
val_cats_dir = os.path.join(validation_dir, 'cats')
val_dogs_dir = os.path.join(validation_dir, 'dogs')
train_total = len(os.listdir(train_cats_dir)) + len(os.listdir(train_dogs_dir))
val_total = len(os.listdir(val_cats_dir)) + len(os.listdir(val_dogs_dir))
print(train_total, val_total)
image_height, image_width = 150, 150
batch_size = 16
epochs= 15

#inputs = keras.Input(shape=(image_width, image_height, 3))
vggmodel = keras.applications.VGG16(include_top=False, weights = 'imagenet', input_shape=(image_width, image_height, 3))
print(type(vggmodel))
model = keras.Sequential()
for layer in vggmodel.layers:
    model.add(layer)
model.summary()



'''last_layer = vggmodel.get_layer('block5_pool').output
x = keras.layers.Flatten(name='flatten')(last_layer)
x = keras.layers.Dense(256, activation='relu', name='dense1')(x)
x = keras.layers.Dropout(0.5, name='drop')(x)
out = keras.layers.Dense(1, activation='sigmoid', name='output')(x)
model = keras.Model(vggmodel.input, out)'''
top_model = keras.Sequential()
top_model.add(keras.layers.Flatten(input_shape=model.output_shape[1:]))
top_model.add(keras.layers.Dense(256, activation='relu'))
top_model.add(keras.layers.Dropout(0.5))
top_model.add(keras.layers.Dense(1, activation='sigmoid'))
#top_model.load_weights('top_model_weights.h5')
model.add(top_model)

for layer in model.layers[:-1]:
    layer.trainable =False

model.summary()
model.compile(optimizer='adam' , loss='binary_crossentropy', metrics=['accuracy'])
train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data = train_data_gen.flow_from_directory(train_dir, target_size=(image_height,image_width), class_mode='binary', batch_size=batch_size)
val_data = val_data_gen.flow_from_directory(validation_dir, target_size=(image_height, image_width), class_mode='binary', batch_size=batch_size)

model.fit(train_data, epochs=epochs, steps_per_epoch=train_total//batch_size, validation_data=val_data, validation_steps=val_total//batch_size)
model.save_weights("vgg16finetune_weights.h5")
model.save('vgg16finetune.h5')

