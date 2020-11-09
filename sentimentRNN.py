import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import numpy as np
np.random.seed(12)
tf.random.set_seed(3)
imdb = tf.keras.datasets.imdb
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=80000)
voc_size = 80000
word_index = imdb.get_word_index()
word_index = {k : v+3 for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict((v,k) for (k,v) in word_index.items())
if __name__ == '__main__':
    def decode(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])
    #text = decode(train_x[0])

    train_data = keras.preprocessing.sequence.pad_sequences(train_x, maxlen=250, padding='post', value=word_index['<PAD>'])
    test_data = keras.preprocessing.sequence.pad_sequences(test_x, maxlen=250, padding='post', value=word_index['<PAD>'])

    val_data = train_data[:5000]
    train_data = train_data[5000:]
    val_y = train_y[:5000]
    train_y = train_y[5000:]


    maxlen = 250
    print(train_data)
    embed_dim = 32
    epochs = 5


    inputs = keras.Input(shape=train_data.shape[1:])
    x = keras.layers.Embedding(voc_size, embed_dim, input_length=maxlen)(inputs)
    x = keras.layers.LSTM(64)(x)
    outputs = keras.layers.Dense(1)(x)

    imodel = keras.Model(inputs, outputs)
    imodel.summary()

    model = keras.Sequential()
    model.add(keras.layers.Embedding(voc_size, embed_dim, input_length=maxlen))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_data, train_y, epochs=epochs, validation_data=(val_data, val_y))
    model.save("sentimentRNN.h5")
    model = keras.models.load_model('sentimentRNN.h5')
    test_review = np.array([test_data[0]])
    prediction = model.predict(test_review)
    decoded = decode(test_data[0])
    print(decoded, '\n', prediction)
    print(model.output.shape)



