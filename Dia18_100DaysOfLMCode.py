
from keras.datasets import imdb
import keras as kr
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Usage of imdb dataset which contains the content from 50000 reviews: 25000 positives and 25000 negatives.
# Each sample contains indexes corresponding to dictionary words, and we will select (therefore keep in the
# model) the 10000 more popular ones for the training
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000
) # One of the optional arguments, index_from, is 3 by default, so some code will subtract 3 from indexes

## How can we train a model that, given a number of words, classifies the review into positive or negative?
## With a hot encoder, for example: we establish 10000 classes with values 0 or 1 (do not appear or do so)

# First we will check the words of a review using the dictionary that relates the words and indexes
# using the first element of the training data
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # indexes as keys

# "?" As signal of the index not existing, and for us an indication that is the start of the sentence,
# since the codification is 0: "padding," 1: "start of sequence," 2: "unknown."
sample_phrase = " ".join([reverse_word_index.get(i-3, "?") for i in train_data[0]])

# Let's define a multihot encoder, since we are not using sklearn
def multihot_encode(sequences, num_classes):
    array = np.zeros((len(sequences), num_classes))
    for i, sequence in enumerate(sequences):
        array[i][sequence] = 1.
    return array

x_train = multihot_encode(train_data, num_classes = 10000)
x_test = multihot_encode(test_data, num_classes = 10000)
y_train, y_test = train_labels.astype("float32"), test_labels.astype("float32") # floats are faster

# We are going to apply 3 layers (further explanation on the choice of layers in a certain problem
# in the next chapter, for now it is like 90% faith) as follows:
model = kr.Sequential(
    [
        layers.Dense(16, activation = "relu"), # relu is just applied for extend the hypothesis space
                                               # to non-linear transformations in 16 dim spaces
        layers.Dense(16, activation = "relu"),
        layers.Dense(1, activation = "sigmoid"),
    ]
)
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy", # Works well for probabilities, since we apply sigmoid in the last layer
    metrics = ["accuracy"],
)

## To validate the data (we must check our loss values and metrics at different epochs) we will separate
## the training set into 2
x_valid = x_train[:10000]
x_partial = x_train[10000:]
y_valid = y_train[:10000]
y_partial = y_train[10000:]

history = model.fit(
    x_partial,
    y_partial,
    epochs = 20,
    batch_size = 512,
    validation_data = (x_valid, y_valid),
)
# We could have used validation_split = 0.2 instead of defining the validation and partial sets
#
# Let's check the values obtained for each epoch:
history_dict = history.history
print(history_dict.keys())

fig, axes = plt.subplots(nrows = 1, ncols = 2)
epochs = np.linspace(1,20,20)

axes[0].plot(epochs, history_dict["loss"], label = "Training Loss")
axes[0].plot(epochs, history_dict["val_loss"], label = "Validation Loss")
axes[1].plot(epochs, history_dict["accuracy"], label = "Training Accuracy")
axes[1].plot(epochs, history_dict["val_accuracy"], label = "Validation Accuracy")
axes[0].set_title("Loss Function Value (train vs val)")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss Function Value")
axes[0].set_xticks(epochs)
axes[0].legend()
axes[1].set_title("Accuracy (train vs val)")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].set_xticks(epochs)
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"The minimum value for the loss function at validation data is reached "
      f"at {int(np.argmin(history_dict['val_loss']))+1} epochs.")
print(f"The maximum value for the accuracy at validation data is reached "
      f"at {int(np.argmax(history_dict['val_accuracy']))+1} epochs.")

# We can clearly see how the model after the fourth epoch started overfitting to the data. Therefore,
# our best guess for the number of epochs chosen to evaluate the model in test samples is around 3 or 4,
# the minimum value for loss and maximum value for accuracy
#
# Let's build our model from scratch using 4 epochs
model = kr.Sequential(
    [
        layers.Dense(16, activation = "relu"),
        layers.Dense(16, activation = "relu"),
        layers.Dense(1, activation = "sigmoid"),
    ]
)
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)
model.fit(x_train, y_train, epochs = 4, batch_size = 512)
results = model.evaluate(x_test, y_test) # loss value and selected metrics (accuracy)
probs = model.predict(x_test) # probabilites of the review being positive

print(f"The results from our model with 4 epochs are a loss value of {results[0]:.3f} and "
      f"an accuracy of {results[1]:.3f}.")