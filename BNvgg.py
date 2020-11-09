import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Activation, GlobalAveragePooling2D
from keras import backend as K
import keras

def conv_block(units, activation='relu', dropout=0.2, block=1, layer=1):

    def layer_names(input):
        x = Conv2D(units, (3,3), padding='same', name='block{}_conv{}'.format(block, layer))(input)
        x = BatchNormalization(name="block{}_bn{}".format(block, layer))(x)
        x = Activation(activation, name="block{}_act{}".format(block, layer))(x)
        x = Dropout(dropout, name="block{}_drop{}".format(block, layer))(x)
        return x
    return layer_names

def dense_block(units, activation='relu', dropout=0.5, name='fc1' ):

    def layer_names(input):
        x = Dense(units, name=name)(input)
        x = BatchNormalization(name='{}_bn'.format(name))(x)
        x = Activation(activation, name='{}_ac'.format(name))(x)
        return x
    return layer_names

def VGG(input_tensor=None, input_shape=None, classes=1000, dropout=0.3, conv_dropout=0.1, activation='relu'):


    img_input = Input(shape=input_shape) if input_tensor is None else (Input(tensor=input_tensor, shape=input_shape) if not K.is_keras_tensor(input_tensor) else input_tensor)

    x = conv_block(64, activation=activation,dropout=conv_dropout, block=1, layer=1)(img_input)
    x = conv_block(64, activation=activation,dropout=conv_dropout, block=1, layer=2)(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pooling')(x)

    x = conv_block(128,activation=activation, dropout=conv_dropout, block=2, layer=1)(x)
    x = conv_block(128,activation=activation, dropout=conv_dropout, block=2, layer=2)(x)
    x = MaxPooling2D((2,2), strides=(2, 2), name='block2_pool')(x)

    x = conv_block(256,activation=activation, dropout=conv_dropout, block=3, layer=1)(x)
    x = conv_block(256,activation=activation, dropout=conv_dropout, block=3, layer=2)(x)
    x = conv_block(256,activation=activation, dropout=conv_dropout, block=3, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = conv_block(512,activation=activation, dropout=conv_dropout, block=4, layer=1)(x)
    x = conv_block(512,activation=activation, dropout=conv_dropout, block=4, layer=2)(x)
    x = conv_block(512,activation=activation, dropout=conv_dropout, block=4, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = conv_block(512,activation=activation, dropout=conv_dropout, block=5, layer=1)(x)
    x = conv_block(512,activation=activation, dropout=conv_dropout, block=5, layer=2)(x)
    x = conv_block(512,activation=activation, dropout=conv_dropout, block=5, layer=3)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = GlobalAveragePooling2D()(x)
    #x = keras.layers.Flatten()(x)

    x = dense_block(4096, activation=activation, dropout=dropout, name='fc1')(x)
    x = dense_block(4096, activation=activation, dropout=dropout, name='fc2')(x)

    x = Dense(classes, activation='softmax', name='output')(x)

    inputs = input_tensor if input_tensor is not None else img_input

    return keras.Model(inputs, x)



model = VGG(input_shape=(224, 224, 3))
model.summary()