# Minimalistic and naive promoter predictor.

# importing the required packages
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, LayerNormalization, \
    Add, Embedding, Layer, Reshape, Dropout, \
    Dense, Conv1D, Input, Flatten, Activation, BatchNormalization

# length of the promoter sequences
input_size = 201
# positive set from EPD, sequences are one-hot encoded, eg A = [True, False, False, False]
positive = utils.parse_fasta("hg38_prom.fa")
# Random ACGT sequences
negative = utils.random_dna(len(positive), input_size)
# Positive labels
ones = np.ones(len(positive))
# Negatives labels
zeros = np.zeros(len(negative))

# Full data is concatenation of positive and negative set
X_data = np.asarray(positive + negative)
Y_data = np.concatenate([ones, zeros])

# Splitting the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.1, random_state=0)

# Deep learning model definition
# Input layer
inputs = Input(shape=(input_size, 4), dtype=tf.float32)
x = inputs
# Dense layer (Fully connected) with 128 neurons and activation relu
x = Dense(128, activation="relu")(x)
# More advanced convolutional layer, which solves the problem faster
# x = Conv1D(128, kernel_size=7, strides=1, activation="relu")(x)
# Flatten layer to get 1D output from the previously 2D shape. See the model summary.
x = Flatten()(x)
# Output layer with only one neuron, 1 for promoter, 0 for non-promoter.
outputs = Dense(1, activation="relu")(x)

# Creating the model object which can be trained and used for prediction.
our_model = Model(inputs, outputs, name="our_model")
print(our_model.summary())

# Compiling the model, we chose the loss function for training and the algorithm (ADAM)
our_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

# Training the model on our promoter data in batches of 16 sequences for 10 iterations.
our_model.fit(X_train, y_train, batch_size=16, epochs=10)

# Predicting the test set and printing the AUC
pred = our_model.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
auc = metrics.auc(fpr, tpr)
print(auc)

