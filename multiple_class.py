from sklearn import model_selection
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import sklearn
img_width, img_height = 224, 224


image = 'cat.jpg'
x = keras.preprocessing.image.load_img(image)
x = Image.Image.resize(x, (img_width, img_height))
data = keras.preprocessing.image.img_to_array(x)
print(data.shape)
data = np.expand_dims(data, axis=0)
print(data.shape)
data = keras.applications.imagenet_utils.preprocess_input(data)
#print(data, type(data))

PATH = os.path.join(r'C:\Users\shast\.keras\datasets', 'cats_and_dogs_filtered')
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, "dogs")
val_cats_dir = os.path.join(validation_dir, 'cats')
val_dogs_dir = os.path.join(validation_dir, 'dogs')
image_data = []

for data in os.listdir(train_dir):
    img_list = os.listdir(train_dir+'/'+data)
    for image in img_list:
        img_path = train_dir + '/' + data + '/' + image
        img = keras.preprocessing.image.load_img(img_path, target_size=(img_width, img_height))
        x = np.asarray(img)
        x = keras.applications.imagenet_utils.preprocess_input(x)
        #print('input image shape', x.shape)
        image_data.append(x)

image_data = np.array(image_data)

print(image_data.shape)

names = ['cat', 'dog']
num_samples = image_data.shape[0]
labels = np.ones((num_samples), dtype = 'int64')

labels[0:1000] = 0
labels[1000:2000] = 1

#labels = keras.utils.to_categorical(labels, 2)


x,y = sklearn.utils.shuffle(image_data, labels, random_state =2)
#print(type(x), type(y))

'''for i in range(5):
    plt.imshow(x[i+100])
    plt.xlabel(names[y[i+100]])
    plt.show()
print(y)'''
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, y, test_size= 0.2, random_state=2)
Y_train = tf.convert_to_tensor(Y_train)
Y_test = tf.convert_to_tensor(Y_test)
#print(Y_train.shape, Y_test.shape, X_test.shape)

image_input = keras.layers.Input(shape=(img_width, img_height, 3))
print(type(image_input))
vgg_model = keras.applications.vgg16.VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
#vgg_model.summary()

last_layer = vgg_model.get_layer('fc2').output
out = keras.layers.Dense(1, activation='sigmoid', name='output')(last_layer)
model = keras.Model(image_input, out)


for layer in model.layers[:-1]:
    layer.trainable = False
model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
hist = model.fit(X_train, Y_train, epochs=12, batch_size= 32, validation_data=(X_test, Y_test))
model.save('ft.h5')