import utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, \
    Add, Embedding, Layer, Reshape, Dropout, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization

input_size = 201
positive = utils.parse_fasta("hg38_prom.fa")
negative = utils.random_dna(len(positive), 201)
ones = np.ones(len(positive))
zeros = np.zeros(len(negative))
X_data = np.asarray(positive + negative)
Y_data = np.concatenate([ones, zeros])

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1, random_state=0)

input_shape = (input_size, 4)
inputs = Input(shape=input_shape, dtype=tf.float32)
x = inputs
x = Conv1D(128, kernel_size=7, strides=1, name="pointwise", activation="relu")(x)
x = Flatten()(x)
outputs = Dense(1, activation="sigmoid")(x)

our_model = Model(inputs, outputs, name="our_model")
print(our_model.summary())

our_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

our_model.fit(X_train, y_train, batch_size=16, epochs=10)

pred = our_model.predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
auc = metrics.auc(fpr, tpr)
print(auc)

