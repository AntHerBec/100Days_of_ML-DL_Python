
import keras as kr
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data.reshape((60000, 28*28)).astype("float32") / 255
test_data = test_data.reshape((10000, 28*28)).astype("float32") / 255
num_epochs = 10
batchsize = 128

# Now the data is all between 0 and 1, so we can easily get random noise from a uniform [0,1] distribution
white_noise_images = np.random.random(train_data.shape)
black_images = np.zeros(train_data.shape)

data_plus_noise = np.concatenate([train_data, white_noise_images], axis = 1)
data_plus_black = np.concatenate([train_data, black_images], axis = 1)

def get_model():
    model = kr.Sequential(
        [
            layers.Dense(512, activation = "relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation = "softmax"),
        ]
    )
    model.compile(
        optimizer = "adam",
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"],
    )
    return model

history_noise = get_model().fit(
    data_plus_noise,
    train_labels,
    epochs = num_epochs,
    batch_size = batchsize,
    validation_split = 0.2,
)
history_black = get_model().fit(
    data_plus_black,
    train_labels,
    epochs = num_epochs,
    batch_size = batchsize,
    validation_split = 0.2,
)

epochs = np.linspace(1, num_epochs, num_epochs)
fig, axes = plt.subplots(nrows = 1, ncols = 2)

axes[0].plot(epochs, history_noise.history["val_accuracy"], "r--", label = "Random Noise")
axes[0].plot(epochs, history_black.history["val_accuracy"], "b:", label = "Black Region")
axes[0].set_title("Validation Accuracy for Modified Data")
axes[0].set_xlabel("Number of Epochs")
axes[0].set_ylabel("Validation Accuracy")
axes[1].plot(epochs, history_noise.history["val_loss"], "r--", label = "Random Noise")
axes[1].plot(epochs, history_black.history["val_loss"], "b:", label = "Black Region")
axes[1].set_title("Validation Loss Value for Modified Data")
axes[1].set_xlabel("Number of Epochs")
axes[1].set_ylabel("Validation Loss Value")

plt.legend()
plt.show()

# Both the loss value and accuracy are notably worse on the random-noise-added sample, since it leads to
# more overfitting that the addition of the black region does not cause (it's the same for all labels)
#
# Since the model will always find a way to fit the model if it has access to enough parameters, is there
# any certainty that the results from training a model will generalize instead of just memorizing inexisting
# rules, like a huge dictionary that accumulates all samples would do?
#
# The "manifold hypothesis" explains why it makes sense to train our data as long as there is (mostly) valid
# data that real-world data sets, usually with very high dimensions/sizes, in this case 256*784 (possible
# values per pixel * number of pixels) concentrates on far smaller manifolds embedded in that space, like the
# one containing the distinguishable digits on the mnist dataset, which are smooth and restricted so that
# moving along the manifolds implies logical, progressive changes that transform each digit into another
#
# Being able to classify brand new data by relating it to inputs in the manifold by interpolating (not in a
# linear way, but as intermediate points in the manifold) and not having to work with the whole input space,
# just this so-called "latent" manifolds, is what justifies the generalized use of deep learning




