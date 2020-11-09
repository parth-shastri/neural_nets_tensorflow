import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
import tensorflow_datasets as tfds
from IPython import display



dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']


BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256

def random_jitter(img):

    img = tf.image.resize(img, size=[286, 286])

    img = tf.image.random_crop(img, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    if tf.random.uniform(()) > 0.5:

        img = tf.image.random_flip_left_right(img)

    return img

def normalize(img):

    img = tf.cast(img, tf.float32)
    img = (img/127.5) - 1

    return img

def preprocess_train(img, label):

    img = normalize(img)
    img = random_jitter(img)

    return img

def preprocess_test(img, label):

    img = normalize(img)

    return img

train_horses = train_horses.map(preprocess_train,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_zebras = train_zebras.map(preprocess_train,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_horses = test_horses.map(preprocess_test,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_zebras = test_zebras.map(preprocess_test,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(train_horses, train_zebras)

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_dx = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_dy = pix2pix.discriminator(norm_type='instancenorm', target=False)

LAMBDA = 10

sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

plt.subplot(121)
plt.imshow(discriminator_dx(sample_horse)[0, ..., -1], cmap='RdBu_r')
plt.subplot(122)
plt.imshow(discriminator_dy(sample_zebra)[0, ..., -1], cmap='RdBu_r')
plt.show()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# discriminator loss

def disc_loss(real, gen):

    loss_1 = loss_object(tf.ones_like(real), real)
    loss_2 = loss_object(tf.zeros_like(gen), gen)

    total_loss = loss_1 +loss_2

    return total_loss

def generator_loss(gen):

    return loss_object(tf.ones_like(gen), gen)

def cycle_loss(real, cycled):
    loss = tf.reduce_mean(tf.abs(real - cycled))

    return LAMBDA*loss*0.5


def id_loss(real, same):

    loss = tf.reduce_mean(tf.abs(real - same))

    return loss*LAMBDA*0.5

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = 'cycleGAN_ckpt'
checkpoint_prefix = 'ckpt'
checkpoint = tf.train.Checkpoint(generator_g=generator_g,
                                 generator_f=generator_f,
                                 discriminator_dy=discriminator_dy,
                                 discriminator_dx=discriminator_dx,
                                 generator_g_optimizer=generator_f_optimizer,
                                 generator_f_optimizer=generator_f_optimizer,
                                 discriminator_y_optimizer=discriminator_y_optimizer,
                                 discriminator_x_optimizer=discriminator_x_optimizer)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print('Latest checkpoint restored!')


@tf.function

def train_step(real_x, real_y):

    with tf.GradientTape(persistent=True) as tape:

        gen_y = generator_g(real_x, training=True)
        cycled_x = generator_f(gen_y, training=True)

        gen_x = generator_f(real_y, training=True)
        cycled_y = generator_g(gen_x, training=True)

        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_gen_y = discriminator_dy(gen_y, training=True)
        disc_gen_x = discriminator_dx(gen_x, training=True)
        disc_real_y = discriminator_dy(real_y, training=True)
        disc_real_x = discriminator_dx(real_x, training=True)

        disc_loss_x = disc_loss(disc_real_x, disc_gen_x)
        disc_loss_y = disc_loss(disc_real_y, disc_gen_y)

        gen_loss_f = generator_loss(gen_x)
        gen_loss_g = generator_loss(gen_y)

        cycle_loss_x = cycle_loss(real_x, cycled_x)
        cycle_loss_y = cycle_loss(real_y, cycled_y)

        total_cycle_loss = cycle_loss_x + cycle_loss_y

        total_g_loss = gen_loss_g + total_cycle_loss + id_loss(real_y, same_y)
        total_f_loss = gen_loss_f + total_cycle_loss + id_loss(real_x, same_x)

    generator_g_gradients = tape.gradient(total_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_f_loss, generator_f.trainable_variables)
    disc_dx_gradients = tape.gradient(disc_loss_x, discriminator_dx.trainable_variables)
    disc_dy_gradients = tape.gradient(disc_loss_y, discriminator_dy.trainable_variables)

    #gradients -2- optimizer

    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(disc_dx_gradients, discriminator_dx.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(disc_dy_gradients, discriminator_dy.trainable_variables))


def fit(train_data_x, train_data_y, epochs):

    for epoch in range(epochs):

        start = time.time()
        n = 0
        display.clear_output(wait=True)
        print('Epoch : {}'.format(epoch+1))
        for batch_x, batch_y in tf.data.Dataset.zip((train_data_x, train_data_y)):

            train_step(batch_x, batch_y)
            if n % 10 == 0:

                print('.')
                n += 1
            print('\r', end='')

        if (epoch+1) % 5 == 0:

            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for {} epoch is {}'.format(epoch+1, time.time() - start))



EPOCHS = 40

#fit(train_horses, train_zebras, EPOCHS)













