
import keras as kr
from keras import layers
import numpy as np
import os
from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf

### Sequential API: it is the easiest and simplest way of building models. It works just like a list, piling
### layers one after another, but can only have 1 input layer and one output and are applied in a sequence
'''
model = kr.Sequential(
[
    layers.Dense(64, activation = "relu"),
    layers.Dense(10, activation = "softmax"),
]
)

or

model = kr.Sequential()
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10, activation = "softmax"))
'''
'''
model = kr.Sequential()
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10, activation = "softmax"))

print(model.weights) # shows an empty list, no build method applied yet
model.build(input_shape = (None,3))
print(model.weights) # Now the model expects samples of shape (3,) with whatever batch size (None)
'''
## To display the contents of the model, we can use summary
'''
model.summary()
'''
# To name the layers, use argument "name =" in Sequential and the layer type function (in this case Dense)

## To check the contents during the building of the machine, if we know the input shape, we can give it as
## argument for the add function, just as the layers
'''
model = kr.Sequential(name = "Model (Sequential)")
model.add(kr.Input(shape = (3,)))
model.add(layers.Dense(64, activation = "relu", name = "Layer 1 (relu)"))
model.summary()
model.add(layers.Dense(10, activation = "softmax", name = "Layer 2 (softmax)"))
model.summary()
'''

### Functional API: In this approach, the different elements of the model are treated like functions,
### and the model is built from the previously defined inputs and outputs (or its structures)
'''
inputs = kr.Input(shape = (3,), name = "Input")
features = layers.Dense(64, activation = "relu", name = "Layer 1 (relu)")(inputs)
outputs = layers.Dense(10, activation = "softmax", name = "Layer 2 (softmax)")(features)
model = kr.Model(inputs = inputs, outputs = outputs, name = "Model (Functional)")

print(inputs.shape) # First index, None, refers to free value for batch size
print(features.shape)
print(outputs.shape)
model.summary()
'''
# Let's consider the following example:
# -> Goal: building a system to rank customer support tickets by priority and route them to the appropriate
# department.
# -> 3 inputs: Ticket title (text), Ticket body (text), tags added by the user (categorical).
# -> 2 outputs: Priority score (0 or 1) and department that should handle the ticket (softmax).

## Building the model
'''
vocabulary_size = 10000 # the number of different words in the text inputs data
num_tags = 100 # Variables for the one-hot encode
num_departments = 4 # Options for the softmax

# Inputs
title = kr.Input(shape = (vocabulary_size,), name = "Title Input")
body = kr.Input(shape = (vocabulary_size,), name = "Text Body Input")
tags = kr.Input(shape = (num_tags,), name = "Tags Input")
# Features
features = layers.Concatenate()([title, body, tags])
features = layers.Dense(64, activation = "relu", name = "Layer_1_Relu")(features)
# Outputs
priority = layers.Dense(1, activation = "sigmoid", name = "Priority_Output")(features)
department = layers.Dense(num_departments, activation = "softmax", name = "Department_Output")(features)
# Model
model = kr.Model(
inputs = [title, body, tags],
outputs = [priority, department],
name = "Model",
)

## Training the model (important: the order of inputs and outputs must be kept or the lists
# given as dictionaries with the referred name

num_samples = 1280

# Adequate datasets for each input
title_data = np.random.randint(0, 2, size = (num_samples, vocabulary_size))
body_data = np.random.randint(0, 2, size = (num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size = (num_samples, num_tags))
#Adequate datasets for each output
priority_data = np.random.random(size = (num_samples, 1))
department_data = np.random.randint(0, num_departments, size = (num_samples, 1))
# Fitting the model to the data
model.compile(
optimizer = "adam",
loss = ["mean_squared_error", "sparse_categorical_crossentropy"],
metrics = [["mean_absolute_error"], ["accuracy"]],
)
# If you don't want to keep the order:
# model.compile(
#    optimizer = "adam",
#    loss = {"priority": "mean_squared_error", "department": "sparse_categorical_crossentropy"},
#    metrics = {"priority": ["mean_absolute_error"], "department": ["accuracy"]},
#)
# And the same applies to the rest of steps (fit, evaluate and predict)
model.fit(
[title_data, body_data, tags_data],
[priority_data, department_data],
epochs=1,
)
model.evaluate(
[title_data, body_data, tags_data],
[priority_data, department_data],
)
priority_preds, department_preds = model.predict(
[title_data, body_data, tags_data]
)

# With functional API it is easy to plot graphs that really help to understand the model structure

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'   # Necessary for plotting the following
                                                                       # graph (graphviz, shown on the left)
kr.utils.plot_model(model, show_shapes = True, show_layer_names = True)

print(model.layers)
print(model.layers[3].input, model.layers[3].output)

# It is also possible to use intermediate features for updating the model, like getting new outputs
# Here it is an example that uses softmax to classify the trouble of solving a ticket in 3 difficulties

features = model.layers[4].output # Output of rely layer
difficulty = layers.Dense(3, activation = "softmax", name = "Difficulty")(features)
new_model = kr.Model(
inputs = [title, body, tags],
outputs = [priority, department, difficulty],
name = "New_Model",
)
kr.utils.plot_model(new_model, to_file = "new_model.png", show_shapes = True, show_layer_names = True)

### Subclassing Model class: a lot of freedom, since you can create whatever structure supported by your call
### function, like loops or recursive layer application, stuff that a graph cannot express correctly. However,
### the chances of making a mistake are larger, and you cannot access the information like before since the
### layers structure is in the call method

class TicketModel(kr.Model):

def __init__(self, num_departments):
    super().__init__()
    self.concat = layers.Concatenate()
    self.relu = layers.Dense(64, activation = "relu")
    self.priority_score = layers.Dense(1, activation = "sigmoid")
    self.department_class = layers.Dense(num_departments, activation = "softmax")

def call(self, inputs):
    title = inputs["title"]
    body = inputs["body"]
    tags = inputs["tags"]

    features = self.concat([title, body, tags])
    features = self.relu(features)
    priority = self.priority_score(features)
    department = self.department_class(features)

    return priority, department

model = TicketModel(num_departments = 4)
priority, department = model(
{"title": title_data, "body": body_data, "tags": tags_data}
)
'''
# The different ways to create a model (Sequential, Functional, Subclassing) can be mixed together,
# as shown by the 2 examples below. This allows for more interesting and efficient models
class Classifier(kr.Model):
    def __init__(self, num_classes = 2):
        super().__init__()
        if num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = num_classes
            activation = "softmax"
        self.dense = layers.Dense(num_units, activation = activation)

    def call(self, inputs):
        return self.dense(inputs)

inputs = kr.Input(shape = (3,))
features = layers.Dense(64, activation = "relu")(inputs)
outputs = Classifier(num_classes = 10)(features)
model = kr.Model(inputs = inputs, outputs = outputs)


inputs = kr.Input(shape=(64,))
outputs = layers.Dense(1, activation="sigmoid")(inputs)
binary_classifier = kr.Model(inputs=inputs, outputs=outputs)

class MyModel(kr.Model):
    def __init__(self, num_classes = 2):
        super().__init__()
        assert num_classes == 2
        self.dense = layers.Dense(64, activation = "relu")
        self.classifier = binary_classifier

    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)

model = MyModel()

def get_mnist_model():
    inputs = kr.Input(shape = (28*28,))
    features = layers.Dense(512, activation = "relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation = "softmax")(features)
    model = kr.Model(inputs = inputs, outputs = outputs)
    return model

(images, labels), (test_images, test_labels) = mnist.load_data()
images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]



# Subclasses for other objects, like metrics, might be needed if they are not implemented in Keras
# In that case, we have to define our own update method
class RootMeanSquareError(kr.metrics.Metric):

    def __init__(self, name = "rmse", **kwargs): # Creates the original weights (as zero), and refers to
                                                 # the parent class by super.__init__ (the Metric class)
        super().__init__(name = name, **kwargs)
        self.mse_sum = self.add_weight(name = "mse_sum", initializer = "zeros")
        self.total_samples = self.add_weight(name = "total_samples", initializer = "zeros")

    # Next, we update the weights given the predicted and true labels, and calculate the value of metrics
    def update_state(self, y_true, y_pred, sample_weight = None):
        y_true = kr.ops.one_hot(y_true, num_classes = kr.ops.shape(y_pred)[1]) # turns categorical data into one-hot
        mse = kr.ops.sum(kr.ops.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = kr.ops.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        return kr.ops.sqrt(self.mse_sum / self.total_samples)

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0.)
'''
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy", RootMeanSquareError()],  # applying this mew metric
)
model.fit(
    train_images,
    train_labels,
    epochs=3,
    validation_data=(val_images, val_labels), # I guess it comes with the get_mnist_model
)
test_metrics = model.evaluate(test_images, test_labels)
'''
### Callbacks: Objects that are passed to the model (fit() call) and are frequently called during training.
### They are very versatile: checkpoints, early stopping after detecting problems, adjusting the value of some
### parameters during training (like learning rate), visualizing certain aspects (like the logging bars in fit()), etc

'''
----------------------------------------
Some built-in callbacks:
----------------------------------------
keras.callbacks.ModelCheckpoint
keras.callbacks.EarlyStopping
keras.callbacks.LearningRateScheduler
keras.callbacks.ReduceLROnPlateau
keras.callbacks.CSVLogger
'''

# A good example of callbacks is the combination of EarlyStopping and ModelCheckpoint, to save the models until the
# improvement by increasing epochs disappears in a specified file

callbacks = [
    kr.callbacks.EarlyStopping(
        monitor = "accuracy",
        patience = 1,               # Interrupts after no improvement for more than 1 epoch
    ),
    kr.callbacks.ModelCheckpoint(
        filepath = "checkpoint_path.keras",
        monitor = "val_loss",
        save_best_only = True,     # Only keeps the one epoch number with the best val_loss value
    )
]

# Now it is only necessary to give the callback list to the argument "callbacks" in fit()
#
# It is possible to save models after training like follows:
# model = keras.models.load_model("checkpoint_path.keras")
#
## You can also implement your own callbacks as subclasses of keras.callbacks.Callback:

class LossHistory(kr.callbacks.Callback):

    def on_train_begin(self, logs):   # logs is a dictionary containing info from the previous patch/epoch/training
        self.per_patch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_patch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(
            range(len(self.per_patch_losses)),
            self.per_patch_losses,
            label = "Training loss per batch",
        )
        plt.xlabel(f"Batch (epoch {epoch+1})")
        plt.ylabel("Loss function value")
        plt.legend()
        plt.savefig(f"Plot_epoch_{epoch+1}", dpi = 300)
        self.per_patch_losses = []
'''
model = get_mnist_model()
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_images,
    train_labels,
    epochs=10,
    callbacks=[LossHistory()],
    validation_data=(val_images, val_labels),
)
'''

### Check cap 7 if TensorBoard is appealing in some moment. For now it is not

### If fit() method is not useful (for example with generative learning or self-supervised learning) we can use
### our own training and evaluation loops. To create a custom training step, we can code like follows:

model = get_mnist_model()
loss_func = kr.losses.SparseCategoricalCrossentropy()
optimizer = kr.optimizers.Adam()

def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training = True)
        loss =  loss_func(targets, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply(gradients, model.trainable_weights)
    return loss

batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]
loss = training_step(inputs = inputs, targets = targets)

# After creating the training_step function, a loop can give the wanted results. However, to avoid writing A LOT of
# code, we can mix the fit() method with a customized training_step by subclassing a Model class and overwriting
# (defining into de class) the train_step method, and also give information to the class to use when compiling

loss_tracker = kr.metrics.Mean(name="loss")

class CustomModel(kr.Model):
    # Overrides the train_step() method
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            # We use self(inputs, training=True) instead of
            # model(inputs, training=True) since our model is the class
            # itself.
            predictions = self(inputs, training=True)
            loss = loss_func(targets, predictions)
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)

        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}

    # Listing the loss tracker metric in the model.metrics property
    # enables the model to automatically call reset_state() on it at
    # the start of each epoch and at the start of a call to evaluate()
    # — so you don't have to do it by hand. Any metric you would like
    # to reset across epochs should be listed here.
    @property
    def metrics(self):
        return [loss_tracker]

# Which, after giving proper inputs and outputs (like in get_mnist_model) is prepared for the fit method with
# the compiled training data, and posterior evaluation, etc

## To get info for the compile method, we can use the following example as reference

class CustomModel(kr.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            # Computes the loss via self.compute_loss
            loss = self.compute_loss(y=targets, y_pred=predictions)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply(gradients, self.trainable_weights)

        # Updates the model's metrics, including the one that tracks the loss
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(targets, predictions)
        # Returns a dict mapping metric names to their current value
        return {m.name: m.result() for m in self.metrics}

def get_custom_model():
    inputs = kr.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(
        optimizer=kr.optimizers.Adam(),
        loss=kr.losses.SparseCategoricalCrossentropy(),
        metrics=[kr.metrics.SparseCategoricalAccuracy()],
    )
    return model

model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)
results = model.evaluate(test_images, test_labels)
print(f"loss = {results[0]:.3f}, accuracy = {results[1]:.3f}")