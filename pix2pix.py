import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from keras import layers
import os
import time

path = r"C:\Users\shast\.keras\datasets"
path = os.path.join(path, 'base')

test_image_paths = os.listdir(path)

train_dir = 'pix2pix_images/train'
val_dir = 'pix2pix_images/val'


BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

#test_data
target_image_paths = [os.path.join(path, image) for image in test_image_paths if image.endswith('.jpg')]
input_image_paths = [os.path.join(path, image) for image in test_image_paths if image.endswith(".png")]

train_image_paths = [os.path.join(train_dir, path) for path in os.listdir(train_dir)]
val_image_paths = [os.path.join(val_dir, path) for path in os.listdir(val_dir)]

print(len(train_image_paths), len(val_image_paths))

test_data = tf.data.Dataset.from_tensor_slices((input_image_paths, target_image_paths))



def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  real_image = image[:, :w, :]
  input_image = image[:, w:, :]

  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def load_test(input_image_path, real_image_path):

    input_image = tf.io.read_file(input_image_path)
    input_image = tf.io.decode_jpeg(input_image)
    input_image = tf.image.resize(input_image, size=[IMG_HEIGHT, IMG_WIDTH])

    real_image = tf.io.read_file(real_image_path)
    real_image = tf.io.decode_png(real_image)
    real_image = tf.image.resize(real_image, size=[IMG_HEIGHT, IMG_WIDTH])

    return input_image, real_image

def random_jitter(input_image, real_image):

    input_image = tf.image.resize(input_image, size=[286, 286])
    input_image = tf.image.random_crop(input_image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    real_image = tf.image.resize(real_image, size=[286, 286])
    real_image = tf.image.random_crop(real_image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.random_flip_left_right(input_image)
        real_image = tf.image.random_flip_left_right(real_image)

    return input_image, real_image


def train_preprocess(input_image_path):

    input_image, real_image = load(input_image_path)

    input_image =(input_image - 127.5)/127.5 #Normalize
    real_image = (real_image - 127.5)/127.5

    input_image, real_image = random_jitter(input_image, real_image)

    return input_image, real_image

def test_preprocess(input_image_path):

    input_image, real_image = load(input_image_path)

    input_image =(input_image - 127.5)/127.5 #Normalize
    real_image = (real_image - 127.5)/127.5

    return input_image, real_image

im1, im2 = test_preprocess(val_image_paths[2])

print(im1.shape, im2.shape)
'''plt.figure()
plt.subplot(121)
plt.imshow(im1)
plt.subplot(122)
plt.imshow(im2)
plt.show()'''

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths))
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths))
val_dataset = val_dataset.map(test_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.shuffle(BUFFER_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)

print(train_dataset.element_spec, '\n', val_dataset.element_spec)

def downsample(filters, size, batch_norm=True):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(layers.Conv2D(filters, size, strides=2,
                             padding='same', kernel_initializer=initializer, use_bias=False))

    if batch_norm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())

    return result

def upsample(filters, size, dropout=False):

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()

    result.add(layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same', kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())

    if dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result

def Generator():

    initializer = tf.random_normal_initializer(0., 0.02)

    inputs = layers.Input(shape=[256, 256, 3])

    down_stack = [

        downsample(64, 4, batch_norm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4)


    ]
    up_stack = [
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),


    ]

    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, use_bias=False,
                                  activation='tanh'
                                  )
    x = inputs

    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


generator = Generator()
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"
#tf.keras.utils.plot_model(generator,to_file='pix2pix_gen.png', show_shapes=True, dpi=64)

def Discriminator():

    initializer = tf.random_normal_initializer(0., 0.02)

    input = layers.Input(shape=(256, 256, 3))

    target = layers.Input(shape=(256, 256, 3))

    concat = layers.concatenate([input, target])

    x = downsample(64, 4, batch_norm=False)(concat)
    x = downsample(128, 4, batch_norm=True)(x)
    x = downsample(256, 4, batch_norm=True)(x)

    zero_pad = layers.ZeroPadding2D()(x)
    conv = layers.Conv2D(512, 4, kernel_initializer=initializer, use_bias=False)(zero_pad)
    x = layers.BatchNormalization()(conv)
    x = layers.ReLU()(x)

    zero_pad = layers.ZeroPadding2D()(x)
    conv = layers.Conv2D(1, 4, kernel_initializer=initializer, use_bias=False)(zero_pad)

    return tf.keras.Model([input, target], conv)


discriminator = Discriminator()

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, gen_output):

    loss1 = loss_object(tf.ones_like(real_output), real_output)
    loss2 = loss_object(tf.zeros_like(gen_output), gen_output)
    total_loss = loss1 + loss2

    return total_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def generate_image(model, input_image, real_image):

    prediction = model(input_image, training=False)
    display_list = [input_image[0], real_image[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']


checkpoint_dir = 'pix2pix_ckpt'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(gen_optim=generator_optimizer, disc_optim=discriminator_optimizer,
                                 generator=generator, discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)


@tf.function
def train_step(input_image, real_image):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, real_image], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, real_image)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

def fit(train_data, epochs, test_data):

    for epoch in range(epochs):

        start = time.time()
        display.clear_output(wait=True)

        for input_image, real_image in test_data.take(1):
            generate_image(generator, input_image, real_image)

        print('Epoch : ', epoch+1)
        for n, (input_image, real_image) in enumerate(train_data):
            print('.', end='')

            if (n+1)%100 == 0:
                print()

            train_step(input_image, real_image)
        print()

        if (epoch + 1) % 20 == 0:

            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))


    checkpoint.save(file_prefix=checkpoint_prefix)



EPOCHS = 150

#fit(train_dataset, EPOCHS, val_dataset)


checkpoint.restore(os.path.join(checkpoint_dir, 'ckpt-8'))
for input_image, real_image in val_dataset.take(1):
    print(input_image)

    predictions = generator(input_image, training=False)
    plt.figure()
    plt.subplot(131)
    plt.imshow(predictions[0])
    plt.title('PREDICTED')
    plt.subplot(132)
    plt.imshow(real_image[0])
    plt.title('REAL')
    plt.subplot(133)
    plt.imshow(input_image[0])
    plt.title("INPUT")
    plt.show()































    










