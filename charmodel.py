import tensorflow as tf
import keras
import numpy as np
import os

data = open('shakespeare.txt', 'r').read()
char = list(set(data))
char = sorted(char)
data_size = len(data)
vocab_size = len(char)
print("The data has ", data_size, "characters, the vocab size is", vocab_size)
char_index = {ch: i for i, ch in enumerate(char)}
index_char = {i: ch for i, ch in enumerate(char)}

data2int = [char_index[c] for c in data]
data2int = np.array(data2int)

seq_length = 100

X = []
y = []

for i in range(data_size // (seq_length + 1)):
    idx = i * (seq_length + 1)
    seq = data2int[idx:idx + seq_length + 1]
    a = seq[:-1]
    b = seq[1:]
    X.append(a)
    y.append(b)
X = np.array(X)
y = np.array(y)
length = len(X)


class seq_generator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=False, shape=(100)):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = shape
        self.on_epoch_end()

    def __len__(self):
        return len(self.x)//(self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_x, temp_y):
        x = np.zeros((self.batch_size, 100))
        y = np.zeros((self.batch_size, 100))

        for i, text in enumerate(temp_x):
            x[i] = text
        for i, text in enumerate(temp_y):
            y[i] = text

        return x, y

    def __getitem__(self, idx):
        i = idx * self.batch_size
        indexes = self.indexes[i: i + self.batch_size]
        temp_x = [self.x[i] for i in indexes]
        temp_y = [self.y[i] for i in indexes]
        #temp_x = np.array(temp_x)
        #temp_y = np.array(temp_y)

        x, y = self.__data_generation(temp_x, temp_y)

        return x, y


dataset = seq_generator(X, y, batch_size=64, shuffle=True)

embed_dim = 256
lstm_units = 1024
BATCH_SIZE = 64

def buildmodel(vocab_size, embed_dim, lstm_units, batchsize):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, embed_dim, batch_input_shape=(batchsize, None)))
    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(lstm_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    model.add(keras.layers.Dense(vocab_size))
    return model

model = buildmodel(vocab_size, embed_dim, lstm_units, BATCH_SIZE)
model.summary()

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
callbacks = [keras.callbacks.ModelCheckpoint('char_model1_checkpoints/ch1_{epoch}', save_weights_only=True)]
history = model.fit(dataset, epochs=25, callbacks=callbacks, steps_per_epoch=length//BATCH_SIZE)

model = buildmodel(vocab_size, embed_dim, lstm_units, batchsize=1)
model.load_weights(tf.train.latest_checkpoint('char_model1_checkpoints'))
model.build(tf.TensorShape([1, None]))
model.summary()

def generate_text(model, string):
    num_generate = 1000
    input = [char_index[i] for i in string]
    input = np.expand_dims(input, 0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input = tf.expand_dims([predicted_id], 0)

        text_generated.append(index_char[predicted_id])

    return (string + ''.join(text_generated))

print(generate_text(model, string=u"ROMEO: "))










