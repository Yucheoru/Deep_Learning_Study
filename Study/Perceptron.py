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

# # What is perceptron?
# Perceptron is an algorithm that receives a number of signals as inputs and outputs one signal.

# # Perceptron's parameters
# * Weights(w) : Eigenvalues multiplied by input signals when sent to neurons(Compared to current, corresponds to resistance)
# * Thresholds($\theta$) : The limits of the sum of signals to output 1.
# * Bias(b) : The values that allows the perceptron to make adjustments to its output independently of the inputs.
#
# # Perceptron's expression
# $ y = \left\{\begin{matrix}
# 0 (b + w_{1}x_{1} + w_{2}x_{2} \leq 0) \\
# 1 (b + w_{1}x_{1} + w_{2}x_{2} >  0)\end{matrix}\right.$

# # Implementation of AND, NAND, OR gates using perceptron algorithm

# +
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

print("AND(0, 0):", AND(0, 0))
print("AND(0, 1):", AND(0, 1))
print("AND(1, 0):", AND(1, 0))
print("AND(1, 1):", AND(1, 1))

# +
import numpy as np

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# +
import numpy as np

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# -

# # The limit of perceptron and its solution
# * Single-layer perceptron only can express a straight-line domain => can't implement XOR gates
# * Through multi-layer perceptron, non-line domain can be expressed => can implement XOR gates

# # XOR gates using multi-layer perceptron

# +
import numpy as np

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print("XOR(0, 0):", XOR(0, 0))
print("XOR(0, 1):", XOR(0, 1))
print("XOR(1, 0):", XOR(1, 0))
print("XOR(1, 1):", XOR(1, 1))
