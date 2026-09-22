
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


tf.ones(shape = (2,1)) # Creates tensors filled with ones
tf.zeros(shape = (2,2)) # Creates tensors filled with zeros
tf.constant([1,5,29]) # Creates tensors from python/numpy ones
tf.random.normal(shape = (3,3), mean = 5, stddev = 0.5) # Creates tensors with random values for distributions
                                                        # such as normal, uniform, gamma, poisson ...

# tensorflow tensors are not assignable, so the following won't work
'''
x = tf.zeros(shape = (4,3))
x[0,0] = 1.0
'''
# We must define an object from class tf.Variable and its method assign

x = tf.Variable(initial_value = tf.zeros(shape = (4,3)))
x.assign(tf.random.normal(shape = (4,3)))
x[0,0].assign(25.)


# The defined operations assign_add and assign_sub are used for adding and substracting tensors to the variable
'''
x.assign_add(tf.ones(shape = (4,3)))
x.assign_sub(tf.ones(shape = (4,3)))
'''
# Apart from the assignments, there are a lot of operations: tf.square, tf.sqrt, summing tensors element-wise (+),
# tf.matmul, tf.concat((a,b), axis = _ ), tf.nn.relu ...
#
# Now we can begin with the implementation of gradients using GradientTape(). Let's consider an example
time = tf.constant(0.0) # scalar tensor
with tf.GradientTape() as outer_tape: # starts a "with" block to record the information and names it outer_tape
                                      # Useful since errors in the block will automatically end the recording
    with tf.GradientTape() as inner_tape:
        inner_tape.watch(time)        # time is a constant tensor, so we need to tell the tape what to focus on.
        outer_tape.watch(time)        # If time was a tf.Variable, this step would not be necessary
        position = 4.9 * time ** 2
    speed = inner_tape.gradient(position, time)
acceleration = outer_tape.gradient(speed, time)

print(speed) # Result: 0.0, multiplied by time
print(acceleration) # Result: 9.8, constant, gravity acceleration on earth

# Finally, let's try a real example of linear classification
nsamples_class = 1000
neg_samples = np.random.multivariate_normal(
    mean = [0,2], cov = [[1,0.5],[0.5,1]], size = nsamples_class
)
pos_samples = np.random.multivariate_normal(
    mean = [-2,0], cov = [[1,0.5],[0.5,1]], size = nsamples_class
)
inputs = np.vstack((neg_samples, pos_samples)).astype("float32")
targets = np.vstack(
    (
        np.zeros((nsamples_class, 1), dtype = "float32"),
        np.ones((nsamples_class, 1), dtype = "float32"),
    )
)
# As we want a linear classifier, the prediction will only need a single layer without activation

input_dim = 2
output_dim = 1
W = tf.Variable(initial_value = tf.random.normal(shape = (input_dim, output_dim)))
print(W)
b = tf.Variable(initial_value = tf.zeros(shape = (output_dim,)))

def model(inputs, W, b):  # The model just applies an affine transformation
    return tf.matmul(inputs, W) + b

def mse(targets, predictions):  # Calculates the mean squared error of the predictions
    losses = tf.square(targets - predictions)
    return tf.reduce_mean(losses) # gives the mean of losses as a scalar

learning_rate = 0.1

@tf.function(jit_compile = True) # Faster calculation using the mask
def trainingstep(inputs, targets, W, b):
    with tf.GradientTape() as tape:
        prediction = model(inputs, W, b)
        loss = mse(targets, prediction)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(grad_loss_wrt_W * learning_rate)
    b.assign_sub(grad_loss_wrt_b * learning_rate)
    return loss

i = 0
prev_loss = 0
for step in range(1, 101):
    loss = trainingstep(inputs, targets, W, b)
    dif = abs(prev_loss - loss)
    tol = 1e-4
    i += 1
    if i % 10 == 0 or dif <= tol:
        print(f"Loss at step {step}: {loss:.4f}")
        if dif <= tol:
            print(f"The difference in the loss function between this step and "
                  f"the previous one was {dif:0.5f}")
            break
    prev_loss = loss

prediction = model(inputs, W, b)
print(f"The model equation after training was loss = ({float(W[0,0]):0.3f}, "
      f"{float(W[1,0]):0.3f})·x + {float(b[0]):0.5f}")
plt.scatter(inputs[:, 0], inputs[:, 1], c = prediction[:, 0] > 0.5) # CLass "1" when prediction > 0.5
x = np.linspace(-10,10,1001)
y = - W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, c = "r")
plt.xlim(min(inputs[:,0]) - 1, max(inputs[:,0]) + 1)
plt.ylim(min(inputs[:,1]) - 1, max(inputs[:,1]) + 1)
plt.show()
