
# Lesson 2


### Sigmoid

A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve. Often, sigmoid function refers to the special case of the logistic function shown in the first figure and defined by the formula.

Sigmoid Functions are used excessively in neural networks. What distinguishes the perceptron from sigmoid neuron or logistic neuron is the presence of the sigmoid function or the logistic function in the sigmoid neuron.

On one hand, the perceptron outputs discrete 0 or 1 value, a sigmoid neuron outputs a more smooth or continous range of values between 0 and 1.

### Softmax
Any time we wish to represent a probability distribution over a discrete variable with **n** possible values, we may use the softmax function. This can be seen as a generalization of the sigmoid function which was used to represent a probability distribution over a binary variable.

Softmax functions are most often used as the output of a classifier, to represent the probability distribution over **n** different classes. More rarely, softmax functions can be used inside the model itself, if we wish the model to choose between one of
**n** different options for some internal variable.

### One-hot encoding

Sometimes data might be consisting of variables, such as 0, and 1 denoting a git, and no gift classes, however sometimes we have more classes, for example a Duck, a Beaver and Walrus, so we might add another class for example 2, but no this will assume class dependency between data, so we do the encoding of class availability for each record as shown in image below.

![One Hot Encoding](/notes/Lesson-2/images/one_hot_encoding.png)

This is a practice of building good representations of data in hand.

Another good example is the following, an example of generating one-hot vector representations for words using a simple document

![One Hot Encoding](/notes/Lesson-2/images/one_hot_encoding_2.png)

If a document has a vocabulary ```V``` with ```|V|``` words, we can represent the words with
one-hot vectors. In other words, we have ```V -dimensional``` representation vectors, and
we associate each unique word with an index in this vector. To represent unique
word ```wi```, we set the ith component of the vector to be 1, and zero out all of the other
components.



### Maximum Likelihood

### Cross Entropy

### Gradient Descent

Most deep learning algorithms involve optimization of some sort. Optimization refers to the task of either minimizing or maximizing some function ***f(x)*** by altering ***x***.

A simple optimization method in machine learning is gradient descent ```(GD)```. When you take gradient steps with respect to all  mm  examples on each step, it is also called Batch Gradient Descent.

A variant of this is Stochastic Gradient Descent ```(SGD)```, which is equivalent to mini-batch gradient descent where each mini-batch has just 1 example. The update rule that you have just implemented does not change. What changes is that you would be computing gradients on just one training example at a time, rather than on the whole training set.

![Gradient Descent Slide 1](/notes/Lesson-2/images/gradient_descent.png)

The code examples below illustrate the difference between stochastic gradient descent and (batch) gradient descent.

* **(Batch) Gradient Descent**

```python
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost.
    cost = compute_cost(a, Y)
    # Backward propagation.
    grads = backward_propagation(a, caches, parameters)
    # Update parameters.
    parameters = update_parameters(parameters, grads)
```

* **Stochastic Gradient Descent**

```python
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost = compute_cost(a, Y[:,j])
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads)
```

>**Note**:
* The difference between gradient descent, mini-batch gradient descent and stochastic gradient descent is the number of examples you use to perform one update step.
* You have to tune a learning rate hyperparameter  αα .
* With a well-turned mini-batch size, usually it outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large).

> **What you should remember**:

* Shuffling and Partitioning are the two steps required to build mini-batches
* Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.

### Exercise Implementation

#### 1. Gradient Descent

```python
# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def output_formula(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

def error_formula(y, output):
    return - y * np.log(output) - (1 - y) * np.log(1-output)

def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias
```

### Momentum

Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, the direction of the update has some variance, and so the path taken by mini-batch gradient descent will _"oscillate"_ toward convergence. Using momentum can reduce these oscillations.

Momentum takes into account the past gradients to smooth out the update. We will store the 'direction' of the previous gradients in the variable  _**v**_ . Formally, this will be the exponentially weighted average of the gradient on previous steps. You can also think of  _**v**_ as the "velocity" of a ball rolling downhill, building up speed (and momentum) according to the direction of the gradient/slope of the hill.

![Momentum 1](/notes/Lesson-2/images/opt_momentum.png)

>Figure: The red arrows shows the direction taken by one step of mini-batch gradient descent with momentum. The blue points show the direction of the gradient (with respect to the current mini-batch) on each step. Rather than just following the gradient, we let the gradient influence  **_v_**  and then take a step in the direction of  **_v_** .


Note that:

The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and start to take bigger steps.
If **_β_=0**, then this just becomes standard gradient descent without momentum.
How do you choose **_β_**?

The larger the momentum **_β_** is, the smoother the update because the more we take the past gradients into account. But if **_β_** is too big, it could also smooth out the updates too much.
Common values for **_β_** range from 0.8 to 0.999. If you don't feel inclined to tune this,  **_β=0.9_** is often a reasonable default.
Tuning the optimal **_β_** for your model might need trying several values to see what works best in term of reducing the value of the cost function  JJ .
What you should remember:

Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
You have to tune a momentum hyperparameter ββ and a learning rate **_α_**.

### Adam

**_Adam_** is one of the most effective optimization algorithms for training neural networks. It combines ideas from **_RMSProp_** and **_Momentum_**.

Momentum usually helps, but given a small learning rate and a simplistic dataset, its impact is almost negligeable. Also, the huge oscillations you see in the cost come from the fact that some minibatches are more difficult thans others for the optimization algorithm.

Adam on the other hand, clearly outperforms mini-batch gradient descent and Momentum. If you run a model for more epochs on a simple dataset, all three methods will lead to very good results. However, you've seen that Adam converges a lot faster.

>Some advantages of Adam include:

* Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum)
* Usually works well even with little tuning of hyperparameters (except  **_α_** )

### Neural Network Architecture

This section starts by showing how linear models actually form a whole probalistic space, and non-linear models is almost like combining 
these models together (adding it for example).

We can think of it as a linear combination of linear models.

![Gradient Descent Slide 1](/notes/Lesson-2/images/neural_network_architecture_sigmoid.png)



### Regularization

What is L2-regularization actually doing?:

L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

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


### Gradient Checking

>What you should remember from this notebook:

* Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
* Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process.

### Multiple layers

Now, not all neural networks look like the one above. They can be way more complicated! In particular, we can do the following things:

* Add more nodes to the input, hidden, and output layers.
* Add more layers.

![Multi-Layer Neural Network](/notes/Lesson-2/images/neural_network_architecture_layers_2.png)

We can imagine a Neural Network as a set of Linear models combined together to have a non-Linear model, like for example the following image, a 3D space model based on two other models, this intuituin helps understand more NN.

![Multi-Layer NN N-Dimensional Space](/notes/Lesson-2/images/neural_network_architecture_layers_n_dimensional_space.png)

### Multi-class Classification




