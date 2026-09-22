
import keras as kr
import keras.metrics
from keras import layers
from keras.datasets import reuters, california_housing
import numpy as np
import matplotlib.pyplot as plt


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000
) # As in day 18, we load_data has a default argument index_from = 3, so we must consider it to decode the data
word_index = reuters.get_word_index()
reverse_index = dict([(value, key) for (key, value) in word_index.items()])

# Function for decoding, in case it is necessary or interesting to read the reuter sentence
def decode(data_samples):
    decoded = []
    if (isinstance(data_samples[0], list)):
        return np.array([decode(sample) for sample in data_samples])
    text = " ".join([reverse_index.get(i-3, "?") for i in data_samples])
    decoded.append(text)
    return decoded

# Values for unique indexes in labels. We can see there are 46 classes in total
print(np.unique(train_labels))

# First, let's apply a multi hot encoder like in day 18, and manipulate data in a convenient way
def multihot_encode(sequences, num_classes):
    array = np.zeros((len(sequences), num_classes))
    for i, sequence in enumerate(sequences):
        array[i][sequence] = 1.
    return array

# In addition, the labels are not binary anymore, so we have to use the multihot_encode on y (there won't be
# errors since the labels are excludent) or use the one built in keras (here we show both of them)

x_train = multihot_encode(train_data, num_classes = 10000)
x_test = multihot_encode(test_data, num_classes = 10000)
y_train = multihot_encode(train_labels, num_classes = 46)
y_test = multihot_encode(test_labels, num_classes = 46)

# or y_. = to_categorical(._labels), functionin keras.utils

# From now until the end of the multiclass problem modelling, the code will be almost the same as day 18
# except from the choice of the loss function. It will become different when starting with regression
model = kr.Sequential(
    [
        layers.Dense(64, activation = "relu"),
        layers.Dense(64, activation = "relu"),
        layers.Dense(46, activation = "softmax"),
    ]
)
# We will add another metric: the accuracy considering top 3 most classes, to account for the correct label
# having the second or third largest probability instead of only focusing on the first one
top_3_accuracy = keras.metrics.TopKCategoricalAccuracy(k = 3, name = "top_3_accuracy")

model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy", top_3_accuracy],
)
x_valid = x_train[:len(x_train) // 3]
x_partial = x_train[len(x_train) // 3:]
y_valid = y_train[:len(y_train) // 3]
y_partial = y_train[len(y_train) // 3:]

history = model.fit(
    x_partial,
    y_partial,
    epochs = 20,
    batch_size = 512,
    validation_data = (x_valid, y_valid),
)
history_dict = history.history
print(history_dict)
print(history_dict.keys())

fig, axes = plt.subplots(nrows = 1, ncols = 3)
epochs = np.linspace(1,20,20)

axes[0].plot(epochs, history_dict["loss"], label = "Training Loss")
axes[0].plot(epochs, history_dict["val_loss"], label = "Validation Loss")
axes[1].plot(epochs, history_dict["accuracy"], label = "Training Accuracy")
axes[1].plot(epochs, history_dict["val_accuracy"], label = "Validation Accuracy")
axes[2].plot(epochs, history_dict["top_3_accuracy"], label = "Training Top 3 Accuracy")
axes[2].plot(epochs, history_dict["val_top_3_accuracy"], label = "Validation Top 3 Accuracy")
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
axes[2].set_title("Top 3 Accuracy (train vs val)")
axes[2].set_xlabel("Epochs")
axes[2].set_ylabel("Top 3 Accuracy")
axes[2].set_xticks(epochs)
axes[2].legend()
plt.show()

print(f"The minimum value for the loss function at validation data is reached "
      f"at {int(np.argmin(history_dict['val_loss']))+1} epochs.")
print(f"The maximum value for the accuracy at validation data is reached "
      f"at {int(np.argmax(history_dict['val_accuracy']))+1} epochs.")
print(f"The maximum value for the top 3 accuracy at validation data is reached "
      f"at {int(np.argmax(history_dict['val_top_3_accuracy']))+1} epochs.")

# Now the values for maxima or minima differ considerably: they oscillate a lot. Therefore, we could infer
# from the plots that a good epoch number is from 9 to 11, since it is from there the model starts overfitting
#
# Now, we build the model from scratch using 10 as epoch number
model = kr.Sequential(
    [
        layers.Dense(64, activation = "relu"),
        layers.Dense(64, activation = "relu"),
        layers.Dense(46, activation = "softmax"),
    ]
)
model.compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = ["accuracy", top_3_accuracy],
)
model.fit(
    x_train,
    y_train,
    epochs = 20,
    batch_size = 512,
)

results = model.evaluate(x_test, y_test)
prediction = model.predict(x_test) # probabilities (softmax) for each class

print(f"The results from our model with 10 epochs are a loss value of {results[0]:.3f}, "
      f"an accuracy of {results[1]:.3f} and a top 3 accuracy of {results[2]:.3f}.")

## It is possible to analyse all of this using integer tensors as labels (just the way they were provided)
## using sparce_categorical_crossentropy as loss function


'''
INFO FROM THE DATASET:
The longitude and latitude of the approximate geographic center of the area.
The median age of houses in the district.
The population of the district. The districts are pretty small: the average population is 1,425.5.
The total number of households.
The median income of those households.
The total number of rooms in the district, across all homes located there. Typically in the low thousands.
The total number of bedrooms in the district.
'''
(train_data, train_targets), (test_data, test_targets) = (
    california_housing.load_data(version="small")
)

print(train_data.shape, test_data.shape)
print(train_targets.shape, test_targets.shape)
print(train_data[:10], train_targets[:10])

# Since the data values are very different and the labels huge (large weights), we will scale them
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
max_target = max(train_targets)

x_train, x_test = (train_data - mean) / std, (test_data - mean) / std
y_train, y_test = train_targets / max_target, test_targets / max_target

# The number of samples is small, so we are going to use a small model (2 layers with 64 units) to avoid
# overfitting, which is more usual in small datasets

def get_model():

    model = kr.Sequential(
        [
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer = "adam",
        loss = "mean_squared_error",
        metrics = ["mean_absolute_error"],
    )
    return model

# Again, since the samples are very few, we will apply k_fold variation to get the different scores for
# each number of epochs, varying the validation data and taking the average of the individual scores

k = 4
num_val = len(x_train) // k
full_scores = []
for i in range(k):

    print(f"Processing fold #{i+1}")

    x_val = x_train[i * num_val:(i + 1) * num_val]
    y_val = y_train[i * num_val:(i + 1) * num_val]
    x_partial = np.concatenate([x_train[:i * num_val], x_train[(i + 1) * num_val:]], axis = 0)
    y_partial = np.concatenate([y_train[:i * num_val], y_train[(i + 1) * num_val:]], axis = 0)

    model = get_model()
    model.fit(
        x_partial,
        y_partial,
        epochs = 50,
        batch_size = 16,
        verbose = 0,
    )
    scores = model.evaluate(x_val, y_val, verbose = 0)
    full_scores.append(scores[1])

print([round(i, 3) for i in full_scores])
print(round(np.mean(full_scores), 3))
'''
'''
k = 4
num_val = len(x_train) // k
num_epochs = 200
all_errors = []
for i in range(k):

    print(f"Processing fold #{i+1}")

    x_val = x_train[i * num_val:(i + 1) * num_val]
    y_val = y_train[i * num_val:(i + 1) * num_val]
    x_partial = np.concatenate([x_train[:i * num_val], x_train[(i + 1) * num_val:]], axis = 0)
    y_partial = np.concatenate([y_train[:i * num_val], y_train[(i + 1) * num_val:]], axis = 0)

    model = get_model()
    history = model.fit(
        x_partial,
        y_partial,
        validation_data = (x_val, y_val),
        epochs = num_epochs,
        batch_size = 16,
        verbose = 0,
    )
    errors = history.history["val_mean_absolute_error"]
    all_errors.append(errors)

avg_errors = [
    np.mean([x[i] for x in all_errors]) for i in range(num_epochs)
]

epochs = range(1, len(avg_errors) + 1)
plt.plot(epochs, avg_errors)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE (log scale)")
plt.yscale("log")
plt.semilogy()
plt.show()

# We can visually estimate the start of overfitting at around 120 epochs, so let's build the model
# from scratch with this value for num_epochs
model = get_model()
model.fit(x_train, y_train, epochs = 120, batch_size = 16, verbose = 0)
results = model.evaluate(x_test, y_test)

print(f"The results from our model with 120 epochs are a loss value (mse) of {int(results[0]*max_target)} and "
      f"a mae of {int(results[1]*max_target)}.")