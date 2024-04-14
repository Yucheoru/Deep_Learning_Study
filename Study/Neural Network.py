# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # What is Neural Network?
# Neural Network is a model inspired by the neuronal organization found in the biological beural networks in animal brains
#
# ## Components of Neural Network
# * Input-layer :
# * Hidden-layer
# * Output-layer

# ## Activation Function
# Activation Function is a non-linear function to decide thresholds of perceptron
# => To make up Neural Network with multi-layer, activation function must be a non-linear function, because using a linear function as it ignores hidden-layer.

# 1. Sigmoid function
# : A sigmoid function is a bounded, differentiable(미분 가능한), real function that is defined for all real input values and has a non-negative derivative(도함수) at each point and exactly one inflection point(변곡점).

# +
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
# -

# 2. ReLU function
# : an activation function defined as the positive part of its argument

# +
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0, x)
    
x = np.array([-1.0, 1.0, 2.0])
print(ReLU(x))

x = np.arange(-6.0, 6.0, 1.0)
y = ReLU(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()
# -

# ## Implementation of 3-layer Neural Network
#
# 1. Sending signals from input-layer to first layer
# * Input-layer : $X = \begin{pmatrix}
# x_{1} & x_{2} \\
# \end{pmatrix}$
# * Weight : $W^{i} = \begin{pmatrix}
# w_{11}^{i} & w_{21}^{i} & w_{31}^{i} \\
# w_{12}^{i} & w_{22}^{i} & w_{32}^{i} \\
# \end{pmatrix}$
# * Bias : $B^{i} = \begin{pmatrix}
# b_{1}^{i} & b_{2}^{i} & b_{3}^{i} \\
# \end{pmatrix}$
# * Hidden-layer(i-layer) value : $A^{i} = XW^{i} + B^{i} = \begin{pmatrix}
# a_{1}^{i} & a_{2}^{i} & a_{3}^{i} \\
# \end{pmatrix}$

# +
import numpy as np

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
# -

# 2. Sending signals from first layer to second layer by using sigmoid function
# * Hidden-layer(i-layer)value caculated by activation function = $Z^{i}$

# +
Z1 = sigmoid(A1)
print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)


# -

# 3. Sending signals from second layer to output-layer by using identity function

# +
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)


# -

# 4.Total cleanup code

# +
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(X, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)


# -

# ## Activation function in output-layer
# : In the machine learning processes, they are divided into classification and regression. 
# Classification is the process of which the data belongs to.
# Regression is the process of predicting succesive figures from input data.
# Grenerally, identity function is used for regression, and softmax function is used for classification.
#
# ## Softmax function
# : In the output layer using softmax function, since each neuron in the output-layer is affected by all input signals, the output of the softmax function receives all input signals.
# \\
# Softmax function has distintive feature that its output can be interpreted as probability(확률), since its total sum is 1.

def primitive_softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
#This function can evoke overflow, So, another softmax function code is needed.
#To implement new softmax function for prevent overflow, maximum value among input signals is generally needed. 


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


# ## Classification for number of handwriting
# ### Batch
#
# Batch is the data in one package.
# Batch allows the result of the input data to be output at once when numeral data are input at once.
# There are two main reasons why batch processing drastically reduces processing time per data.
# * Most of numpy are optimized to handle large arrays efficiently
# * It reduces the load

# +
import sys, os
sys.path.append(os.pardir) # setting to import files from parent directories
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import  Image

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# return MNIST data into form of "(train_img, train_label), (test_img, test_label)"
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# Batch processing
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# -


