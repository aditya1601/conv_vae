import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from vae import VarAutoEncoder


# Load Data
def preprocess(image, _):
    """Return normalized image for both input and output label"""
    image = tf.cast(image, tf.float32)/255.
    image = tf.where(image < 0.5, 0., 1.)
    return image, image


ds_train, ds_test = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True
)
ds_train = ds_train.map(preprocess).shuffle(1024).batch(64)
ds_test = ds_test.map(preprocess).shuffle(1024).batch(64)


# Train model
model = VarAutoEncoder(latent_dim=2)
model.compile(optimizer=keras.optimizers.Adam(1e-4))
history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=50
)


# Save model
model.save("models/vae-v1")
