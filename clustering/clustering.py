import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Dense, Input, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


def build(input_size, latent_dim):
    l1_weight = 0
    layer_units = [512, 256]
    inputs = Input(shape=input_size)
    x = inputs
    for f in layer_units:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    latent = Dense(latent_dim, use_bias=False, activity_regularizer=regularizers.l1(l1_weight))(x)
    encoder = Model(inputs, latent, name="encoder")
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs
    for f in layer_units[::-1]:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    x = Dense(1)(x)
    outputs = Activation("tanh")(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    return autoencoder


adt = pd.read_csv("GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz", sep=",", skiprows=1, header=None)
rna = pd.read_csv("GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz", sep=",", skiprows=1, header=None)
both = adt.append(rna, ignore_index=True)
features = both.iloc[:, 0]
both.drop(both.columns[0], axis=1, inplace=True)
data = both.to_numpy().T
autoencoder = build(len(features), 64)
autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=1e-4))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
autoencoder.fit(data, data, epochs=100, batch_size=16, validation_split=0.1, callbacks=[callback])
encoder = autoencoder.get_layer("encoder")
latent = encoder.predict(np.asarray(data))
np.savetxt("latent.csv", latent, delimiter=",")


