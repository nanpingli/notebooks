# Save & Restore a Model

Save and Restore a model using TensorFlow v2. In this example, we will go over both low and high-level approaches: 
- Low-level: TF Checkpoint.
- High-level: TF Module/Model saver.

This example is using the MNIST database of handwritten digits as toy dataset
(http://yann.lecun.com/exdb/mnist/).

- Author: Aymeric Damien
- Project: https://github.com/aymericdamien/TensorFlow-Examples/


```python
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
```


```python
# MNIST dataset parameters.
num_classes = 10 # 0 to 9 digits
num_features = 784 # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 50
```


```python
# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255.
```


```python
# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
```

## 1) TF Checkpoint

Basic logistic regression


```python
# Weight of shape [784, 10], the 28*28 image features, and total number of classes.
W = tf.Variable(tf.random.normal([num_features, num_classes]), name="weight")
# Bias of shape [10], the total number of classes.
b = tf.Variable(tf.zeros([num_classes]), name="bias")

# Logistic regression (Wx + b).
def logistic_regression(x):
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(tf.matmul(x, W) + b)

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)
```


```python
# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = logistic_regression(x)
        loss = cross_entropy(pred, y)

        # Compute gradients.
        gradients = g.gradient(loss, [W, b])

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, [W, b]))
```


```python
# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
```

    step: 50, loss: 535.380981, accuracy: 0.656250
    step: 100, loss: 354.681152, accuracy: 0.765625
    step: 150, loss: 225.300934, accuracy: 0.785156
    step: 200, loss: 163.948761, accuracy: 0.859375
    step: 250, loss: 129.653534, accuracy: 0.878906
    step: 300, loss: 170.743576, accuracy: 0.859375
    step: 350, loss: 97.912575, accuracy: 0.910156
    step: 400, loss: 144.119141, accuracy: 0.906250
    step: 450, loss: 164.991943, accuracy: 0.875000
    step: 500, loss: 145.191666, accuracy: 0.871094
    step: 550, loss: 82.272644, accuracy: 0.925781
    step: 600, loss: 149.180237, accuracy: 0.878906
    step: 650, loss: 127.171280, accuracy: 0.871094
    step: 700, loss: 116.045761, accuracy: 0.910156
    step: 750, loss: 92.582680, accuracy: 0.906250
    step: 800, loss: 108.238007, accuracy: 0.894531
    step: 850, loss: 92.755638, accuracy: 0.894531
    step: 900, loss: 69.131119, accuracy: 0.902344
    step: 950, loss: 67.176285, accuracy: 0.921875
    step: 1000, loss: 104.205658, accuracy: 0.890625


## Save and Load with TF Checkpoint


```python
# Save weights and optimizer variables.
# Create a dict of variables to save.
vars_to_save = {"W": W, "b": b, "optimizer": optimizer}
# TF Checkpoint, pass the dict as **kwargs.
checkpoint = tf.train.Checkpoint(**vars_to_save)
# TF CheckpointManager to manage saving parameters.
saver = tf.train.CheckpointManager(
      checkpoint, directory="./tf-example", max_to_keep=5)
```


```python
# Save variables.
saver.save()
```




    './tf-example/ckpt-1'




```python
# Check weight value.
np.mean(W.numpy())
```




    -0.09673191




```python
# Reset variables to test restore.
W = tf.Variable(tf.random.normal([num_features, num_classes]), name="weight")
b = tf.Variable(tf.zeros([num_classes]), name="bias")
```


```python
# Check resetted weight value.
np.mean(W.numpy())
```




    -0.0083419625




```python
# Set checkpoint to load data.
vars_to_load = {"W": W, "b": b, "optimizer": optimizer}
checkpoint = tf.train.Checkpoint(**vars_to_load)
# Restore variables from latest checkpoint.
latest_ckpt = tf.train.latest_checkpoint("./tf-example")
checkpoint.restore(latest_ckpt)
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x12cf965d0>




```python
# Confirm that W has been correctly restored.
np.mean(W.numpy())
```




    -0.09673191



## 2) TF Model

Basic neural network with TF Model


```python
from tensorflow.keras import Model, layers
```


```python
# MNIST dataset parameters.
num_classes = 10 # 0 to 9 digits
num_features = 784 # 28*28

# Training parameters.
learning_rate = 0.01
training_steps = 1000
batch_size = 256
display_step = 100
```


```python
# Create TF Model.
class NeuralNet(Model):
    # Set layers.
    def __init__(self):
        super(NeuralNet, self).__init__(name="NeuralNet")
        # First fully-connected hidden layer.
        self.fc1 = layers.Dense(64, activation=tf.nn.relu)
        # Second fully-connected hidden layer.
        self.fc2 = layers.Dense(128, activation=tf.nn.relu)
        # Third fully-connecter hidden layer.
        self.out = layers.Dense(num_classes, activation=tf.nn.softmax)

    # Set forward pass.
    def __call__(self, x, is_training=False):
        x = self.fc1(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

# Build neural network model.
neural_net = NeuralNet()
```


```python
# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    y_true = tf.cast(y_true, tf.int64)
    crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return tf.reduce_mean(crossentropy)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)
```


```python
# Optimization process. 
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = neural_net(x, is_training=True)
        loss = cross_entropy(pred, y)

        # Compute gradients.
        gradients = g.gradient(loss, neural_net.trainable_variables)

        # Update W and b following gradients.
        optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables))
```


```python
# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = neural_net(batch_x, is_training=False)
        loss = cross_entropy(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
```

    step: 100, loss: 2.188605, accuracy: 0.902344
    step: 200, loss: 2.182990, accuracy: 0.929688
    step: 300, loss: 2.180439, accuracy: 0.945312
    step: 400, loss: 2.178496, accuracy: 0.957031
    step: 500, loss: 2.177517, accuracy: 0.968750
    step: 600, loss: 2.177163, accuracy: 0.968750
    step: 700, loss: 2.177454, accuracy: 0.960938
    step: 800, loss: 2.177589, accuracy: 0.960938
    step: 900, loss: 2.176507, accuracy: 0.964844
    step: 1000, loss: 2.177557, accuracy: 0.960938


## Save and Load with TF Model


```python
# Save TF model.
neural_net.save_weights(filepath="./tfmodel.ckpt")
```


```python
# Re-build neural network model with default values.
neural_net = NeuralNet()
# Test model performance.
pred = neural_net(batch_x)
print("accuracy: %f" % accuracy(pred, batch_y))
```

    accuracy: 0.101562



```python
# Load saved weights.
neural_net.load_weights(filepath="./tfmodel.ckpt")
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x13118b950>




```python
# Test that weights loaded correctly.
pred = neural_net(batch_x)
print("accuracy: %f" % accuracy(pred, batch_y))
```

    accuracy: 0.960938

