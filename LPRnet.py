import tensorflow as tf
import os
import numpy as np
from keras import layers as nn
from keras import preprocessing
from keras.preprocessing import image
from keras import backend as K
import matplotlib.pyplot as plt
import keras

data_dir = "LPR/train"
images = sorted(os.listdir(data_dir))
labels = [i.strip('.jpg').strip("- 1").strip('(1)').strip('(2)').strip(' -2').strip(" - 3").strip("_").strip('_0')
                    .strip("_1").strip("_4").strip("_6").strip("_5").strip("_7").strip("_8").strip("_9")
          for i in images]
CHARS = sorted(set(char for label in labels for char in label))
print("Number of images :", len(images))
print("Number of unique characters :", len(CHARS))
print("The characters found :", CHARS)

image_paths = sorted([os.path.join(data_dir, path) for path in images if path.endswith(".jpg")])
print(len(image_paths))
batch_size = 8
img_height = 24
img_width = 94
in_lr = 1e-4
lr = keras.optimizers.schedules.ExponentialDecay(in_lr, decay_steps=500, decay_rate=0.9)


max_length = max([len(i) for i in labels])

char_to_int = {c: i for i, c in enumerate(CHARS)}
int_to_char = dict((i, c) for c, i in char_to_int.items())

def num_to_char(num):
    return ''.join([int_to_char.get(i, '?') for i in num])


_labels = []
for label in labels:
    new = [char_to_int[c] for c in label]
    _labels.append(new)
_labels = keras.preprocessing.sequence.pad_sequences(_labels, maxlen=max_length, padding='post')
print(_labels.shape)

def split_data(images, labels, train_size=0.9, shuffle=True):

    size = len(images)
    indices = np.arange(size)

    if shuffle:
        np.random.shuffle(indices)

    train_samples = int(size*train_size)

    x_train, x_test = images[indices[:train_samples]], images[indices[train_samples:]]
    y_train, y_test = labels[indices[:train_samples]], labels[indices[train_samples:]]

    return x_train, x_test, y_train, y_test

x_train, x_valid, y_train, y_valid = split_data(np.array(image_paths), np.array(_labels))

print(x_train.shape, y_train.shape)


#keras Sequence method
class data_generator(keras.utils.Sequence):

    def __init__(self, image_paths, labels, batch_size, img_height, img_width, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = (img_height, img_width)
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)//self.batch_size

    def __getitem__(self, idx):
        i = idx*self.batch_size
        batch_image_paths = self.image_paths[i : i + self.batch_size]
        batch_labels = self.labels[i : i + self.batch_size]

        x = np.zeros((self.batch_size, img_width, img_height, 3), dtype='float32')
        y = np.zeros((self.batch_size, max_length), dtype='int32')

        for i, path in enumerate(batch_image_paths):
            img = image.load_img(path, target_size=self.img_size)
            img = np.asarray(img, dtype='float32')
            #img = img/255.0
            img = tf.transpose(img, perm=[1, 0, 2])
            img = np.array(img)
            if self.augment:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                img = tf.image.random_brightness(img, 0.2)

            x[i] = img

        for i, label in enumerate(batch_labels):

            y[i] = label

        return {'image': x, "label": y}


train_data = data_generator(x_train, y_train, batch_size, img_height, img_width, augment=False)

valid_data = data_generator(x_valid, y_valid, batch_size, img_height, img_width, augment=False)


class CTCLayer(nn.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = tf.keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def simple_block(x, Cout):

    conv_1_sb = nn.Conv2D(Cout//4, (1, 1), activation='relu')(x)
    conv_2_sb = nn.Conv2D(Cout//4, kernel_size=[3, 1], padding='same', activation='relu')(conv_1_sb)
    conv_3_sb = nn.Conv2D(Cout//4, kernel_size=[1, 3], padding='same', activation='relu')(conv_2_sb)
    conv_4_sb = nn.Conv2D(Cout, kernel_size=[1, 1], activation='relu')(conv_3_sb)
    output = conv_4_sb
    return output


def build_model():

    inputs= tf.keras.Input(shape=(img_width, img_height, 3), name='image')

    labels = tf.keras.Input(shape=(None, ), dtype='float32', name='label')

    c1 = nn.Conv2D(64, kernel_size=[3, 3], padding='same')(inputs)
    m1 = nn.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(c1)
    s1 = simple_block(m1, 128)

    x2 = s1

    m2 = nn.MaxPooling2D((3,3), strides=(2, 1), padding='same')(s1)
    s2 = simple_block(m2, 256)
    s3 = simple_block(s2, 256)

    x3 = s3

    m3 = nn.MaxPooling2D((3,3), strides=(2,1), padding='same')(s3)
    d1 = nn.Dropout(0.5)(m3)

    c2 = nn.Conv2D(256, kernel_size=[4, 1], padding='same', activation='relu')(d1)
    d2 = nn.Dropout(0.5)(c2)

    c3 = nn.Conv2D(len(CHARS)+1, kernel_size=(1, 13), padding='same')(d2)
    x = nn.ReLU()(c3)
    #adding global context to the local features as done in Parsenet
    cx = tf.reduce_mean(tf.square(x))
    x = tf.divide(x, cx)

    x1 = nn.AveragePooling2D(pool_size=(4, 1), strides=(4, 1), padding='same')(inputs)
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.divide(x1, cx1)

    x2 = nn.AveragePooling2D(pool_size=(4, 1), strides=(4, 1), padding='same')(x2)
    cx2 = tf.reduce_mean(tf.square(x2))
    x2 = tf.divide(x2, cx2)

    x3 = nn.AveragePooling2D(pool_size=(2, 1), strides=(2, 1), padding="same")(x3)
    cx3 = tf.reduce_mean(tf.square(x3))
    x3 = tf.divide(x3, cx3)

    x = nn.concatenate([x, x1, x2, x3], 3)
    x = nn.Conv2D(len(CHARS)+1, kernel_size=(1, 1), activation="relu", padding='same')(x)
    logits = tf.reduce_mean(x, 2)


    output = CTCLayer(name='ctc_loss')(labels, logits)



    model = tf.keras.Model([inputs, labels], output)
    optimizer = keras.optimizers.Adam(lr)

    model.compile(optimizer)

    return model



model = build_model()
model.summary()
patience = 10
epochs = 1000
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

history = model.fit(
    train_data,
    validation_data=valid_data, 
    epochs=epochs,

)


model.save("saved_models/LPRnet_1000_epochs[bs=8].h5")
#model.load_weights("saved_models/LPRnet_1000_epochs.h5")

prediction_model = keras.Model(model.get_layer('image').input, model.get_layer('conv2d_15').output)
prediction_model.summary()

def decode_predictions(pred):

    input_length = np.ones(pred.shape[0])*pred.shape[1]

    results = keras.backend.ctc_decode(pred, input_length, greedy=True)[0][0][:, :max_length]

    output_text =[]
    for result in results:
        result = num_to_char(np.array(result))
        output_text.append(result)

    return output_text



pred = prediction_model.predict(valid_data)
pred = tf.reduce_mean(pred, axis=2
                      )

outputs = decode_predictions(pred)
print("Predicted:", outputs[1])
print('Real:', num_to_char(y_valid[1]))




