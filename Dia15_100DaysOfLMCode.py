
# Keras is a great module for deep learning, being compatible with tensorflow, pytorch ...
import keras as kr
from keras.datasets import mnist # Basic dataset with 28x28 pixel black and white images of numbers from 0 to 9
import matplotlib.pyplot as plt

# We define our training and testing variables
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
fig, axes = plt.subplots(nrows = 2, ncols = 2)
for i in range(2):
    for j in range(2):
        axes[i,j].imshow(train_images[i+2*j], cmap = plt.cm.binary)
plt.tight_layout()
plt.show()

# We establish the model (2 different layers with dimensions 512 and 10)
model = kr.Sequential(
    [
        kr.layers.Dense(512, activation = "relu"),
        kr.layers.Dense(10, activation = "softmax"),  # softmax function, e^y_k / sum(e^y_i)
    ]
)

# We define the loss function (to minimize), optimizer and metrics returned by the model
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"],
)

# The model works with data in the [0,1] interval, so we have to rescale the data from its values in [0,255]
# Also, we have to reshape it in a matrix (rank 2 tensor) form, with each row of length 28*28
train_images = train_images.reshape((60000, 28*28)) / 255
test_images = test_images.reshape((10000, 28*28)) / 255

# Now we fit the model; epochs is the amount of times we go through the whole dataset, and batches are the
# pieces of dataset you train your data on (more batches mean more stability of parameters, but more overfitting)
# so the total amount of iterations is n_epochs * n_batches
model.fit(train_images, train_labels, epochs = 5, batch_size = 128)

# Finally for today, we can check the model using data from the test_images
test_digits = test_images[:10]
predictions = model.predict(test_digits)
print(predictions)

# We can see that accuracy went up real quick, ending at around 99%, and loss also decreased in every iteration
# Let's check if the predicted results (the ones that maximize the softmax score) correspond to the real labels
for i in range(10):
    predicted = predictions[i].argmax()
    real = test_labels[i]
    if predicted == real: print("Correct prediction of image {}".format(i))
    else: print("Wrong prediction of image {}".format(i))

# And using the whole testing data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"The accuracy of the model is {test_acc:.4f}")