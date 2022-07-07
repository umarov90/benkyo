import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Dense, Input, Dropout
from tensorflow.keras.models import Model
from scipy.stats import zscore
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from tensorflow.keras import regularizers
matplotlib.use("Agg")


def build(input_size, latent_dim):
    layer_units = [1024, 512]
    inputs = Input(shape=input_size)
    x = inputs
    x = Dropout(0.2)(x)
    for i, f in enumerate(layer_units):
        x = Dense(f, activation="relu")(x)
        if i == 0:
            x = Dropout(0.2)(x)

    latent = Dense(latent_dim, use_bias=False)(x)
    encoder = Model(inputs, latent, name="encoder")
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs
    for i, f in enumerate(layer_units[::-1]):
        x = Dense(f, activation="relu")(x)
        if i == 0:
            x = Dropout(0.2)(x)

    outputs = Dense(input_size)(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    return autoencoder


data = pd.read_csv("a.csv", sep=",", header=None)
data = data.to_numpy()
autoencoder = build(len(data[0]), 50)
# autoencoder = tf.keras.models.load_model("autoencoder.h5")
autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=1e-4))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
autoencoder.fit(data, data, epochs=1000, batch_size=16, validation_split=0.1, callbacks=[callback])
autoencoder.save("autoencoder_a.h5")
encoder = autoencoder.get_layer("encoder")
latent = encoder.predict(np.asarray(data))
np.savetxt("latent_a.csv", latent, delimiter=",", fmt="%s")
latent_vectors = np.loadtxt("latent_a.csv", delimiter=",")

# pca = PCA(n_components=2)
# latent_vectors = pca.fit_transform(latent_vectors)
reducer = umap.UMAP()
latent_vectors = reducer.fit_transform(latent_vectors)
np.savetxt("umap_a.csv", latent_vectors, delimiter=",", fmt="%s")
fig, axs = plt.subplots(1,1,figsize=(8,4))
print("Plotting")
sns.scatterplot(x=latent_vectors[:, 0], y=latent_vectors[:, 1], s=5, alpha=0.2, ax=axs)
axs.set_title("Latent space")
axs.set_xlabel("A1")
axs.set_ylabel("A2")
plt.tight_layout()
plt.savefig("clustering.png")


