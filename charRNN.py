import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

char_dataset = tf.data.Dataset.from_tensor_slices(data2int)


seq_length = 100
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def create_data(seq):
    input = seq[:-1]
    target = seq[1:]
    return input, target

dataset = sequences.map(create_data)
for x,y in dataset.take(2):
    print(x.numpy(), '\n', y.numpy())

BATCH_SIZE = 64
epochs=10
embed_dim = 256
lstm_units = 1024

dataset = dataset.shuffle(10000).batch(BATCH_SIZE, drop_remainder=True)
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
callbacks = [keras.callbacks.ModelCheckpoint('char_model_checkpoints/ch_{epoch}', save_weights_only=True)]
#history = model.fit(dataset, epochs=10, callbacks=callbacks)

print(tf.train.latest_checkpoint('char_model_checkpoints'))


model = buildmodel(vocab_size, embed_dim, lstm_units, batchsize=1)
model.load_weights(tf.train.latest_checkpoint('char_model_checkpoints'))
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
        print(predictions)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions/temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        print(predicted_id)
        input = tf.expand_dims([predicted_id], 0)

        text_generated.append(index_char[predicted_id])

    return (string + ''.join(text_generated))

print(generate_text(model, string=u"ROMEO: "))
















