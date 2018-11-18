
# Lesson 2

### Sigmoid

### Softmax
Any time we wish to represent a probability distribution over a discrete variable with **n** possible values, we may use the softmax function. This can be seen as a generalization of the sigmoid function which was used to represent a probability distribution over a binary variable.

Softmax functions are most often used as the output of a classifier, to represent the probability distribution over **n** different classes. More rarely, softmax functions can be used inside the model itself, if we wish the model to choose between one of
**n** different options for some internal variable.

### One-hot encoding

### Maximum Likelihood

### Cross Entropy

### Gradient Descent

Most deep learning algorithms involve optimization of some sort. Optimization refers to the task of either minimizing or maximizing some function ***f(x)*** by altering ***x***.


### Exercise Implementation

#### 1. Gradient Descent

```python
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

def error_formula(y, output):
    return - y*np.log(output) - (1 - y) * np.log(1-output)

def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias
```
