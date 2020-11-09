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
epochs= 50

def bottleneck():
    datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    model = keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(image_width,image_height,3))
    generator = datagen.flow_from_directory(train_dir, target_size=(image_width,image_height), batch_size=batch_size, class_mode=None, shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, train_total//batch_size)
    np.save(open('bottleneck_train.npy', 'wb'), bottleneck_features_train)

    generator = datagen.flow_from_directory(validation_dir, target_size=(image_width,image_height), batch_size=batch_size, class_mode=None, shuffle=False)
    bottleneck_features_val = model.predict_generator(generator, val_total//batch_size)
    np.save(open('bottleneck_val.npy', 'wb'), bottleneck_features_val)

def top_model():
    train_data = np.load(open('bottleneck_train.npy', 'rb'))
    '''x = np.zeros((int(train_total/2), 1))
    y = np.ones((int(train_total/2), 1))
    train_labels = np.concatenate((x, y))
    print(train_data.shape)
    print(train_labels.shape)'''
    train_labels = np.array([0]*(train_total//2) + [1]*(train_total//2))
    print(train_labels.shape)

    val_data = np.load(open('bottleneck_val.npy', 'rb'))
    '''x = np.zeros((int(val_total / 2), 1))
    y = np.ones((int(val_total / 2), 1))
    val_labels =  np.concatenate((x, y))'''
    val_labels = np.array([0] * (val_total // 2) + [1] * (val_total // 2))


    top_model = keras.Sequential()
    top_model.add(keras.layers.Flatten(input_shape = train_data.shape[1:]))
    top_model.add(keras.layers.Dense(256, activation='relu'))
    top_model.add(keras.layers.Dropout(0.5))
    top_model.add(keras.layers.Dense(1, activation='sigmoid'))

    top_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    top_model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
    top_model.save_weights('top_model_weights.h5')
bottleneck()
top_model()