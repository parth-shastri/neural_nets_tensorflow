import tensorflow as tf
import numpy as np
import keras
from keras.preprocessing import image
from keras import layers
from keras.layers import Conv2D, LSTM, Dropout, Dense
from tensorflow import compat
from matplotlib import pyplot as plt
import os
import pathlib
import PIL.Image

data_dir = "captcha_images_v2"
images = sorted(os.listdir(data_dir))
labels = [img.strip('.png').strip() for img in images]
CHARS = sorted(set(char for label in labels for char in label))
print("Number of images :", len(images))
print("Number of unique characters :", len(CHARS))
print("The characters found :", CHARS)

image_paths = sorted([os.path.join(data_dir, path) for path in images if path.endswith(".png")])
print(len(image_paths))
batch_size = 16
img_height = 50
img_width = 200

downsample_factor =4

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

    def __init__(self, image_paths, labels, batch_size, img_height, img_width, char_to_int):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = (img_height, img_width)

    def __len__(self):
        return len(self.image_paths)//self.batch_size

    def __getitem__(self, idx):
        i = idx*self.batch_size
        batch_image_paths = self.image_paths[i : i + self.batch_size]
        batch_labels = self.labels[i : i + self.batch_size]

        x = np.zeros((self.batch_size, img_width, img_height, 1), dtype='float32')
        y = np.zeros((self.batch_size, max_length), dtype='int32')

        for i, path in enumerate(batch_image_paths):
            img = image.load_img(path, color_mode='grayscale', target_size=self.img_size)
            img = np.asarray(img, dtype='float32')
            img = img.reshape((img_height, img_width, 1))
            img = img/255.0
            img = tf.transpose(img, perm=[1, 0, 2])
            img = np.array(img)
            x[i] = img

        for i, label in enumerate(batch_labels):

            y[i] = label

        return {'image': x, "label": y}


train_data = data_generator(x_train, y_train, batch_size, img_height, img_width, char_to_int)

valid_data = data_generator(x_valid, y_valid, batch_size, img_height, img_width, char_to_int)

#tf.data method
'''char_to_num = tf.keras.layers.experimental.preprocessing(
    vocabulary=list(CHARS), num_oov_indices=0, mask_token=None
)'''

def load(image_path, label):

    img = tf.io.read_file(image_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, dtype='float32')
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])

    return {'image':img, 'label': label}

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        load, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_dataset = (
    valid_dataset.map(
        load, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)
class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

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


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None, ), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(CHARS) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# Get the model
model = build_model()
model.summary()
epochs = 100
early_stopping_patience = 10
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs

)
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
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
outputs = decode_predictions(pred)
print("Predicted:", outputs[1])
print('Real:', num_to_char(y_valid[1]))



















