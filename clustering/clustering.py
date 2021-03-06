import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Dense, Input
from tensorflow.keras.models import Model
from scipy.stats import zscore
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
matplotlib.use("Agg")


def build(input_size, latent_dim):
    layer_units = [512, 256]
    inputs = Input(shape=input_size)
    x = inputs
    for f in layer_units:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    latent = Dense(latent_dim, use_bias=False)(x)
    encoder = Model(inputs, latent, name="encoder")
    latent_inputs = Input(shape=(latent_dim,))
    x = latent_inputs
    for f in layer_units[::-1]:
        x = Dense(f)(x)
        x = LeakyReLU(alpha=0.2)(x)

    outputs = Dense(input_size)(x)
    decoder = Model(latent_inputs, outputs, name="decoder")
    autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    return autoencoder


adt = pd.read_csv("GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz", sep=",", skiprows=1, header=None)
rna = pd.read_csv("GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz", sep=",", skiprows=1, header=None)
both = adt.append(rna, ignore_index=True)
features = both.iloc[:, 0]
print(f"Features: {len(features)}")
print(f"Cells: {len(both.iloc[0, :])}")
both.drop(both.columns[0], axis=1, inplace=True)
both.to_csv("data_merged.tsv", sep='\t')
# both = both.apply(zscore)
# data = both.to_numpy().T
# autoencoder = build(len(features), 128)
# # autoencoder = tf.keras.models.load_model("autoencoder.h5")
# autoencoder.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(lr=1e-4))
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
# autoencoder.fit(data, data, epochs=100, batch_size=16, validation_split=0.1, callbacks=[callback])
# autoencoder.save("autoencoder.h5")
# encoder = autoencoder.get_layer("encoder")
# latent = encoder.predict(np.asarray(data))
# np.savetxt("latent.csv", latent, delimiter=",", fmt="%s")
latent_vectors = np.loadtxt("latent.csv", delimiter=",")


def pcc(a, b):
    v = stats.pearsonr(a.flatten(), b.flatten())[0]
    return 1 - v


# latent_vectors = TSNE(n_components=2, random_state=0, perplexity=5, metric=pcc).fit_transform(latent_vectors)
# pca = PCA(n_components=2)
# latent_vectors = pca.fit_transform(latent_vectors)
reducer = umap.UMAP()
latent_vectors = reducer.fit_transform(latent_vectors)
np.savetxt("umap.csv", latent_vectors, delimiter=",", fmt="%s")
fig, axs = plt.subplots(1,1,figsize=(8,4))
print("Plotting")
sns.scatterplot(x=latent_vectors[:, 0], y=latent_vectors[:, 1], s=5, alpha=0.2, ax=axs)
axs.set_title("Latent space")
axs.set_xlabel("PCA1")
axs.set_ylabel("PCA2")
plt.tight_layout()
plt.savefig("PCA.png")


