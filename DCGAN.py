import tensorflow as tf
import keras
from IPython import display
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from keras import layers
import time


(train_data, train_labels), (_, _) = keras.datasets.mnist.load_data()
train_images = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images-127.5)/127.5  #Normalize the images

buffer_size = 60000
batch_size = 256
lr = 1e-4

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(buffer_size).batch(batch_size)

'''for images in train_dataset.take(1):
    plt.imshow(images[1, :, :, 0], cmap='gray')
    plt.colorbar()
    plt.show()'''

def build_generator():

    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape(target_shape=(7, 7, 256)))
    assert model.output_shape== (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

noise =  tf.random.normal([1, 100])
generator = build_generator()
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis(emit=False)
#plt.show()

def build_discriminator():

    model = keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = build_discriminator()
decision = discriminator(generated_image, training=False)

loss = keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, fake):

    real_loss = loss(tf.ones_like(real), real)
    fake_loss = loss(tf.zeros_like(fake), fake)
    total = real_loss + fake_loss

    return total

def generator_loss(fake):
    return loss(tf.ones_like(fake), fake)


generator_optimizer = keras.optimizers.Adam(lr)
discriminator_optimizer = keras.optimizers.Adam(lr)

checkpoint_dir = 'DCGAN_ckpt'
checpoint_prefix = os.path.join(checkpoint_dir, 'ckpt'
                                )
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('DC_GAN_IMAGES/images_at_epoch{}.png'.format(epoch))
    #plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch+1, seed)


        if (epoch+1) % 15 == 0:
            checkpoint.save(file_prefix=checpoint_prefix)

        print("Time for epoch {} is {} sec".format(epoch+1, time.time() - start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)


train(train_dataset, EPOCHS)





