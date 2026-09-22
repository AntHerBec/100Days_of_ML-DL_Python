
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.utils import image_dataset_from_directory
import pathlib

inputs = keras.Input(shape = (32, 32, 3))
x = layers.Rescaling(1.0/255)(inputs)

## Use of residuals to avoid the accumulated noise/error into layers. This assures having a clear way to
### backpropagate our gradients, allowing for deeper networks without worrying of our model not being able
### to improve due to "vanishing gradients"
def residual_block (x, filters, pooling = False):
    residual = x
    x = layers.Conv2D(filters, 3, activation = "relu", padding = "same")(x)
    x = layers.Conv2D(filters, 3, activation = "relu", padding = "same")(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding = "same")(x)
        # Since we use pooling, the residual has to be strided conv
        residual = layers.Conv2D(filters, 1, strides = 2)(residual)
    elif filters != residual.shape[-1]:
        # In case x didn't have the same number of filters as our output
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    return x

# Now we define the 3 blocks, increasing the number of filters on each one
x = residual_block(x, filters = 32, pooling = True)
x = residual_block(x, filters = 64, pooling = True)
# The last block don't need max pooling, because we will apply an average pooling
x = residual_block(x, filters = 128)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation = "sigmoid")(x)
model = keras.Model(inputs = inputs, outputs = outputs)

model.summary()

#---------------------------------------------------------------------------------------------------------------

### Batch Normalization helps is used to dinamically (employing trainable parameters) normalize the outputs from
### intermediate layers in different batches, so:
###
### -> The outputs follow a normalized shape (if needed/beneficial). It then acts as the typical normalization
### -> Unlike typical normalization, it doesn't harm some non-linear activations (for example, sigmoid behaves
###    linearly when close to 0), allows values far from 0 (if it reduces loss function)
###
### It works as the classic normalization when it benefits the model, and can adapt when it doesn't

x = layers.Conv2D(32, 3, use_bias=False)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)


# Better order than
'''
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.BatchNormalization()(x)
'''
# Since you can have a normalized distribution so relu performs best. Think about the best place to apply it

#---------------------------------------------------------------------------------------------------------------

### Lastly, the implementation of Depthwise Separable Convolutional Networks can help in terms of reducing
### overfitting / easing convergence of the method and (if it isn't because NVIDIA GPUs are not very optimized
### to not-ConvNet models) training faster than usual convolutional networks, due to it training each channel
### independently with a convnet, concatenating them and applying a last convnet (1x1) to mix the output
### channels: it is similar to separate the spatial and channel features learning
#
# The next example will contain all these different methods:

inputs1 = keras.Input(shape = (180, 180, 3))
x = layers.Rescaling(1.0/255)(inputs1)
# We can't assume channel correlation with RGB images, since natural images have proportional changes in the 3
# colors due to the brightness (brighter means bigger values of red, blue and green)
x = layers.Conv2D(filters = 32, kernel_size = 5, use_bias = False)(x)

# Each block has 2 separable 2D convnets with relu as activation
for size in [32, 64, 128, 256, 512]:
    residual = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding = "same", use_bias = False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding = "same", use_bias = False)(x)

    x = layers.MaxPooling2D(3, strides = 2, padding = "same")(x)

    residual = layers.Conv2D(size, 1, strides = 2, padding = "same", use_bias = False)(residual)
    x = layers.add([x, residual])

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)

outputs1 = layers.Dense(1, activation = "sigmoid")(x)
model1 = keras.Model(inputs = inputs1, outputs = outputs1)

# Let's train it and see the results
batch_size = 64
image_size = (180, 180)
new_dir = pathlib.Path("kagglehub/datasets/salader/dogsvscats_subset")

train_dataset = image_dataset_from_directory(
    new_dir/"train", image_size = image_size, batch_size = batch_size
)
val_dataset = image_dataset_from_directory(
    new_dir/"validation", image_size = image_size, batch_size = batch_size
)
test_dataset = image_dataset_from_directory(
    new_dir/"test", image_size = image_size, batch_size = batch_size
)
model1.summary()
model1.compile(
optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="kagglehub/datasets/salader/dogsvscats/cap9_checkpoint.keras",
        save_best_only=True,
        monitor="val_loss",
    ),
]

'''keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
),'''

data_augmentation_layers = [              # It is still a small dataset
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
]

def data_augmentation(images, targets):
    for layer in data_augmentation_layers:
        images = layer(images, training = True)
    return images, targets

train_dataset = train_dataset.map(
    data_augmentation, num_parallel_calls= tf.data.AUTOTUNE
)

history = model1.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=callbacks,
)


acc, val_acc = history.history["accuracy"], history.history["val_accuracy"]
loss, val_loss = history.history["loss"], history.history["val_loss"]
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "r--", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


# Check class names detected in validation
print("Validation classes:", val_dataset.class_names)

# Check validation batch label distribution
for images, labels in val_dataset.take(1):
    print("Sample batch labels:", labels.numpy())

test_model = keras.models.load_model("kagglehub/datasets/salader/dogsvscats/cap9_checkpoint.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy = {test_acc:.3f}; Test loss = {test_loss:.3f}")

test_model = keras.models.load_model("kagglehub/datasets/salader/dogsvscats/model_checkpoints.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy = {test_acc:.3f}; Test loss = {test_loss:.3f}")

test_model = keras.models.load_model("kagglehub/datasets/salader/dogsvscats/model_checkpoint_data_augment.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy = {test_acc:.3f}; Test loss = {test_loss:.3f}")
