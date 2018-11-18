
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

![Gradient Descent Slide 1](/notes/Lesson-2/images/gradient_descent.png)


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

### Neural Network Architecture

This section starts by showing how linear models actually form a whole probalistic space, and non-linear models is almost like combining 
these models together (adding it for example).

We can think of it as a linear combination of linear models.

![Gradient Descent Slide 1](/notes/Lesson-2/images/neural_network_architecture_sigmoid.png)


What is L2-regularization actually doing?:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

### Regularization

#### What you should remember -- the implications of L2-regularization on:

* The cost computation:
  * A regularization term is added to the cost

* The backpropagation function:
  * There are extra terms in the gradients with respect to weight matrices

* Weights end up smaller ("weight decay"):
  * Weights are pushed to smaller values.

### Dropout

#### What you should remember about dropout:

* Dropout is a regularization technique.

* You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.

* Apply dropout both during forward and backward propagation.

* During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. For example, if keep_prob is 0.5, then we will on average shut down half the nodes, so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. Dividing by 0.5 is equivalent to multiplying by 2. Hence, the output now has the same expected value. You can check that this works even when keep_prob is other values than 0.5.

**Note**:

>A common mistake when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training.
Deep learning frameworks like tensorflow, PaddlePaddle, keras or caffe come with a dropout layer implementation. Don't stress - you will soon learn some of these frameworks.
