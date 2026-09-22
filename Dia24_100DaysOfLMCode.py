
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
import keras as kr
from keras import layers
from keras.datasets import mnist
from keras.utils import image_dataset_from_directory
import zipfile
import os, shutil, pathlib
import kagglehub
import tensorflow as tf
import tensorflow_hub as hub

### ConvNets/CNNs are specially useful when classifying visual data, getting notably better results than densely
### connected network, and can obtain good results even from small datasets, which is fantastic since access to big,
### varied datasets are almost reserved for large companies or businesses. However (at least in the following example)
### it is shown that it takes much longer to train the model

# Functional API for CNN

inputs = kr.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = kr.Model(inputs=inputs, outputs=outputs)

model.summary()

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)) # 60000 samples of images in format (height, width, channels)
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.astype("float32") / 255

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy", # integers as categories
    metrics = ["accuracy"],
)
model.fit(train_images, train_labels, epochs = 5, batch_size = 64)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss = {test_loss:.3f}, Test Accuracy = {test_accuracy:.3f}")  # Around 99% accuracy, a 50% improvement
                                                                            # from densely conected network (chap. 2)


### FROM NOW ON THE WRITTEN CODE IS USED TO IMPORT A DATASET FROM KAGGLE. TO USE AS (kind of) AN EXAMPLE ON
### HOW TO IMPORT FUTURE DATASETS, SINCE I HAD ENOUGH PROBLEMS TO MAKE A RESPONSIBLE ADULT CRY IN THEIR SHOWER

# Salader's dogs vs cats dataset is a great one for training neural networks on image classification problems,
# with 2 clearly distinct and recognisable classes. It is the one I will be using for some days from now, or at
# least a subset of the images
'''
path = kagglehub.dataset_download("salader/dogsvscats")
print("Path to dataset files:", path)
'''


orig_dir = pathlib.Path("kagglehub/datasets/salader/dogsvscats")
new_dir = pathlib.Path("kagglehub/datasets/salader/dogsvscats_subset")

'''
category_map = {"cat": "cats", "dog": "dogs"}

def mksubset(name, start, end):

    for category, folder_name in category_map.items():
        dst_dir = new_dir / name / category
        os.makedirs(dst_dir, exist_ok=True)

        fnames = [f"{category}.{i}.jpg" for i in range(start, end)]

        for fname in fnames:
            # Rutas posibles donde puede estar la imagen
            possible_sources = [
                orig_dir/"train"/folder_name/fname,
                orig_dir/"test"/folder_name/fname,
            ]

            # Buscar la ruta real donde existe la imagen
            src_path = next(
                (path for path in possible_sources if path.exists()), None
            )

            if src_path:
                shutil.copyfile(src=src_path, dst=dst_dir / fname)
            else:
                print(f"Advertencia: No se encontró la imagen {fname}")

mksubset("train", start = 0, end = 1000)
mksubset("validation", start = 1000, end = 1500)
mksubset("test", start = 1500, end = 2000)
'''

## With our imported subset, we can start building the model: inputs of 180x180 pixels, RGB and rescaled to work
## with values between 0 and 1; and a binary classification, so a sigmoid function should work well

inputs = kr.Input(shape = (180, 180, 3))

x = layers.Rescaling(1.0/255)(inputs)
x = layers.Conv2D(filters = 32, kernel_size = 3, activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 64, kernel_size = 3, activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 128, kernel_size = 3, activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 256, kernel_size = 3, activation = "relu")(x)
x = layers.MaxPooling2D(pool_size = 2)(x)
x = layers.Conv2D(filters = 512, kernel_size = 3, activation = "relu")(x)

# We flatten the resulting features (height, width, 512) into 1D by averaging so the sigmoid can calculate its value

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation = "sigmoid")(x)
model = kr.Model(inputs = inputs, outputs = outputs)
model.summary()
model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)

# Now we can decode the images into data for our model using the following function from the keras API

batch_size = 64
image_size = (180, 180)
train_dataset = image_dataset_from_directory(
    new_dir/"train", image_size = image_size, batch_size = batch_size
)
val_dataset = image_dataset_from_directory(
    new_dir/"validation", image_size = image_size, batch_size = batch_size
)
test_dataset = image_dataset_from_directory(
    new_dir/"test", image_size = image_size, batch_size = batch_size
)

## This datasets are converted into Dataset objects, with methods as .batch(), .shuffle(), .prefetch() and .map()
for data_batch, labels_batch in train_dataset:
    print("data batch shape:", data_batch.shape)
    print("labels batch shape:", labels_batch.shape)
    break

callbacks = [
    kr.callbacks.ModelCheckpoint(
        filepath="kagglehub/datasets/salader/dogsvscats/model_checkpoints.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]

history = model.fit(
    train_dataset,
    epochs=50,
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

test_model = kr.models.load_model("kagglehub/datasets/salader/dogsvscats/model_checkpoints.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy = {test_acc:.3f}; Test loss = {test_loss:.3f}")


test_model = kr.models.load_model("kagglehub/datasets/salader/dogsvscats/modelo_prueba_checkpoints.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy = {test_acc:.3f}; Test loss = {test_loss:.3f}")
'''
'''
callbacks_prueba = [
    kr.callbacks.ModelCheckpoint(
        filepath="kagglehub/datasets/salader/dogsvscats/modelo_prueba_checkpoints.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
modelo_prueba = kr.Model(inputs = inputs, outputs = outputs)
modelo_prueba.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)

history_prueba = modelo_prueba.fit(
    train_dataset,
    epochs=5,
    validation_data=val_dataset,
    callbacks=callbacks_prueba,
)

acc, val_acc = history_prueba.history["accuracy"], history_prueba.history["val_accuracy"]
loss, val_loss = history_prueba.history["loss"], history_prueba.history["val_loss"]
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


## To improve the results (reducing or postponing the appearance of overfitting) we can use data augmentation, that
## is, increasing the dataset artificially by applying transformations to the data images

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
]

def data_augmentation(images, targets):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images, targets

train_dataset = train_dataset.map(
    data_augmentation, num_parallel_calls= 8
)

inputs = kr.Input(shape=(180, 180, 3))
x = layers.Rescaling(1.0 / 255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)


model_aug = kr.Model(inputs=inputs, outputs=outputs)
model_aug.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
callbacks = [
    kr.callbacks.ModelCheckpoint(
        filepath="kagglehub/datasets/salader/dogsvscats/model_checkpoint_data_augment.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = model_aug.fit(
    train_dataset,
    # Since we expect the model to overfit slower, we train for more
    # epochs.
    epochs=100,
    validation_data=val_dataset,
    callbacks=callbacks,
    batch_size = 32,

test_model = kr.models.load_model("kagglehub/datasets/salader/dogsvscats/model_checkpoint_data_augment.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy = {test_acc:.3f}; Test loss = {test_loss:.3f}")

## Another approach for training with a small dataset, apart from Dropout and augmentation, would be using a
## pretrained model from a much larger dataset which doesn't even have to be directly related (for example, the
## book mentions using a model trained on animals/everyday objects to identify furniture). Let's import one from
## tensorflow-hub



# 1. Load MobileNetV2 base configured specifically for (180, 180, 3)
conv_base = kr.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,            # Exclude classification head
    input_shape=(180, 180, 3)     # Custom resolution
)
conv_base.trainable = False       # Freeze pretrained weights

# 2. Preprocessor layer (manual, unlike in keras-hub)
preprocessor = layers.Rescaling(1.0 / 255.0)

# 3. Functional API matching your original structure
inputs = kr.Input(shape=(180, 180, 3))
x = preprocessor(inputs)
x = conv_base(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = kr.Model(inputs, outputs)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.summary()

callbacks = [
    kr.callbacks.ModelCheckpoint(
        filepath="kagglehub/datasets/salader/dogsvscats/model_checkpoint_data_augment.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]

history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=callbacks,
)

test_model = kr.models.load_model("kagglehub/datasets/salader/dogsvscats/model_checkpoint_data_augment.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy = {test_acc:.3f}; Test loss = {test_loss:.3f}")

# We got a 98.1% accuracy, a really good result, and it could go even further by fine-tuning (that is, changing some
# parameters or characteristics of the training like the learning rate)