import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Dense, Flatten, Reshape
import numpy as np


class Sampling(keras.layers.Layer):
    """Sample *z* from the *z_mean* and *z_logvar* from encoder to input in decoder"""

    def call(self, inputs, **kwargs):
        z_mean, z_logvar = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * epsilon


class Encoder(keras.layers.Layer):
    """Maps MNIST digits to triplet (z_mean, z_logvar, z)"""

    def __init__(self, latent_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.conv1 = Conv2D(32, 3, (2,2), activation='relu')
        self.conv2 = Conv2D(64, 3, (2,2), activation='relu')
        self.flatten = Flatten()
        self.dense3_1 = Dense(latent_dim)
        self.dense3_2 = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        z_mean = self.dense3_1(x)
        z_logvar = self.dense3_2(x)
        z = self.sampling((z_mean, z_logvar))
        return z_mean, z_logvar, z


class Decoder(keras.layers.Layer):
    """Reconstructs the image from latent variable *z*"""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense1 = Dense(7*7*32, activation='relu')
        self.reshape = Reshape((7, 7, 32))
        self.deconv1 = Conv2DTranspose(64, 3, 2, padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(32, 3, 2, padding='same', activation='relu')
        self.out = Conv2DTranspose(1, 3, 1, padding='same')

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return self.out(x)


class VarAutoEncoder(keras.Model):
    """Convolutional Variational AutoEncoder Model for MNIST"""

    def __init__(self, latent_dim, **kwargs):
        super(VarAutoEncoder, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    # TODO : Use TFP library functions
    @tf.function
    def log_normal_pdf(self, z, mean, logvar):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((z - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=1)

    def call(self, inputs, **kwargs):
        z_mean, z_logvar, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Compute loss
        cross_entropy_loss = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(inputs, reconstructed), axis=[1, 2, 3])
        kl_loss = self.log_normal_pdf(z, z_mean, z_logvar) - self.log_normal_pdf(z, 0., 0.)
        total_loss = -tf.reduce_mean(cross_entropy_loss - kl_loss)

        self.add_loss(total_loss)
        return reconstructed
