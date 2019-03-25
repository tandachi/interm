import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from logger import logger
import matplotlib.pyplot as plt

plt.switch_backend('agg') 

shape = (28, 28, 1)
epochs = 10000
batch = 32
save_interval = 100


def generator():
    model = Sequential()
    model.add(Dense(256, input_shape=(100,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28 * 28 * 1, activation='tanh'))
    model.add(Reshape(shape))

    return model


def discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=shape))
    model.add(Dense((28 * 28 * 1), input_shape=shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(int((28 * 28 * 1) / 2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model


def stacked_generator_discriminator(D, G):
    D.trainable = False
    model = Sequential()
    model.add(G)
    model.add(D)
    return model


def plot_images(samples=16, step=0):
    filename = "mnist_%d.png" % step
    noise = np.random.normal(0, 1, (samples, 100))
    images = Generator.predict(noise)
    plt.figure(figsize=(10, 10))

    for i in range(images.shape[0]):
        plt.subplot(4, 4, i + 1)
        image = images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close('all')
    
    
Generator = generator()
Generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5, decay=8e-8))

Discriminator = discriminator()
Discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5, decay=8e-8),
                      metrics=['accuracy'])

stacked_generator_discriminator = stacked_generator_discriminator(Discriminator, Generator)
stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5, decay=8e-8))

(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)


for cnt in range(epochs):

    random_index = np.random.randint(0, len(X_train) - batch / 2)
    legit_images = X_train[random_index: random_index + batch // 2].reshape(batch // 2, 28, 28, 1)
    gen_noise = np.random.normal(0, 1, (batch // 2, 100))
    syntetic_images = Generator.predict(gen_noise)

    x_combined_batch = np.concatenate((legit_images, syntetic_images))
    y_combined_batch = np.concatenate((np.ones((batch // 2, 1)), np.zeros((batch // 2, 1))))

    d_loss = Discriminator.train_on_batch(x_combined_batch, y_combined_batch)

    noise = np.random.normal(0, 1, (batch, 100))
    y_mislabled = np.ones((batch, 1))

    g_loss = stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

    logger.info('epoch: {}, [Discriminator: {}], [Generator: {}]'.format(cnt, d_loss[0], g_loss))

    if cnt % save_interval == 0:
        plot_images(step=cnt)