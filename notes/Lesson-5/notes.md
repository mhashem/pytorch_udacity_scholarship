# Lesson 5: Convolutional Neural Networks

## Outline

// todo

## How Computers interpret images

// todo

## MLP Structure & Class Scores

// todo

## Loss & Optimization

// todo

### ReLu Activation Function

The purpose of an activation function is to scale the outputs of a layer so that they are a consistent, small value. Much like normalizing input values, this step ensures that our model trains efficiently!

A ReLu activation function stands for "rectified linear unit" and is one of the most commonly used activation functions for hidden layers. It is an activation function, simply defined as the positive part of the input, `x`. So, for an input image with any negative pixel values, this would turn all those values to `0`, black. You may hear this referred to as "clipping" the values to zero; meaning that is the lower bound.

![RELU](images/relu.png)


### Cross-Entropy Loss

In the [PyTorch documentation](https://pytorch.org/docs/stable/nn.html#crossentropyloss) , you can see that the cross entropy loss function actually involves two steps:

* It first applies a softmax function to any output is sees
* Then applies [NLLLoss](https://pytorch.org/docs/stable/nn.html#nllloss); negative log likelihood loss

Then it returns the average loss over a batch of data. Since it applies a softmax function, we do not have to specify that in the `forward` function of our model definition, but we could do this another way.

#### Another approach

We could separate the softmax and NLLLoss steps.

* In the `forward` function of our model, we would explicitly apply a softmax activation function to the output, `x`.

```python
 ...
 ...
# a softmax layer to convert 10 outputs into a distribution of class probabilities
x = F.log_softmax(x, dim=1)

return x
```

* Then, when defining our loss criterion, we would apply NLLLoss

```python
# cross entropy loss combines softmax and nn.NLLLoss() in one single class
# here, we've separated them
criterion = nn.NLLLoss()
```

This separates the usual `criterion = nn.CrossEntropy()` into two steps: softmax and NLLLoss, and is a useful approach should you want the output of a model to be class probabilities rather than class scores.

## Defining a Network in Pytorch

#### 1. Import libraries

```python
# import libraries
import torch
import numpy as np
```

#### 2. Load and visualize data

```python
from torchvision import datasets
import torchvision.transforms as transforms

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)
```

Visualizing a batch of Training Data

```python
import matplotlib.pyplot as plt
%matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))
```

View image in more detail

```python
img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
```


#### 3. Define Network Architecture

```python
import torch.nn as nn
import torch.nn.functional as F

## TODO: Define the NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear layer (784 -> 1 hidden node)
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# initialize the NN
model = Net()
print(model)
```

Model architecture

```
Net(
  (fc1): Linear(in_features=784, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=512, bias=True)
  (fc3): Linear(in_features=512, out_features=10, bias=True)
  (dropout): Dropout(p=0.2)
)
```

4. Specify Loss and Optimizer 

It's recommended that you use cross-entropy loss for classification. If you look at the documentation (linked above), you can see that PyTorch's cross entropy function applies a softmax funtion to the output layer *and* then calculates the log loss.

```python
# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = torch.optim.Adam(model.parameters())
```

5. Train Network

```python
# number of epochs to train the model
n_epochs = 50  # suggest training between 20-50 epochs

model.train() # prep model for training

for epoch in range(n_epochs):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for data, target in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*data.size(0)
        
    # print training statistics 
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch+1, 
        train_loss
        ))
```

6. Test the Trained Network

```python
# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval() # prep model for *evaluation*

for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
```

## Image Classification Steps

Each time we have an image classification project the following steps are typically made:

1. Load and Visualize Data
2. Preprocess (Normalize, Transform)
3. Do a research to check if someone has already tackled this problem, and then define the Nerual Network model
4. Choose appropriate Loss & Optimization Algorithms for your model then `Train your model`
5. Save model after each check for Validation - Train diversion
6. Test model

![Image Classification Steps](images/image-classification-steps.png)


## MLPs vs CNNs 

MLPs are good but have no idea about the structure of the input being classified, as they attempt to process an input representing the image as 1-D flattened vector.

CNN truely shines for real world problem, where messy data are the case.

![Flattening](images/mlps-flattening.png)

MLPs | CNNs
-----|-----
Only use **_Fully Connected_** layes (huge number of parameters quickly fast incresing computational complexity) | Use Sparsely connected layers
Only accepts **_vectors as input_**| Accept Matrix as Input

> MLPs Flattening Example

![Flattenning NN](images/mlps-flattening-nn.png)

### Local Connectivity

// todo importatn

### Filters and the Convolutional Neural Network

// todo

### Filters and Edges

// todo

### Frequency in Images

We have an intuition of what frequency means when it comes to sound. High-frequency is a high pitched noise, like a bird chirp or violin. And low frequency sounds are low pitch, like a deep voice or a bass drum. For sound, frequency actually refers to how fast a sound wave is oscillating; oscillations are usually measured in cycles/s (Hz), and high pitches and made by high-frequency waves. Examples of low and high-frequency sound waves are pictured below. On the y-axis is amplitude, which is a measure of sound pressure that corresponds to the perceived loudness of a sound and on the x-axis is time.

![Sound Frequency](images/sound-frequency.png)

(Top image) a low frequency sound wave (bottom) a high frequency sound wave.

#### High and low frequency

Similarly, frequency in images is a **rate of change**. But, what does it means for an image to change? Well, images change in space, and a high frequency image is one where the intensity changes a lot. And the level of brightness changes quickly from one pixel to the next. A low frequency image may be one that is relatively uniform in brightness or changes very slowly. This is easiest to see in an example.

![Image High - Low Frequency](images/image-high-low-frequency.png)

Most images have both high-frequency and low-frequency components. In the image above, on the scarf and striped shirt, we have a high-frequency image pattern; this part changes very rapidly from one brightness to another. Higher up in this same image, we see parts of the sky and background that change very gradually, which is considered a smooth, low-frequency pattern.

High-frequency components also correspond to the edges of objects in images, which can help us classify those objects.

### Filters and OpenCV 

// todo


### Convolutional Layer

#### The Importance of Filters

What you've just learned about different types of filters will be really important as you progress through this course, especially when you get to Convolutional Neural Networks (CNNs). CNNs are a kind of deep learning model that can learn to do things like image classification and object recognition. They keep track of spatial information and learn to extract features like the edges of objects in something called a convolutional layer. Below you'll see an simple CNN structure, made of multiple layers, below, including this "convolutional layer".

![CNN Layers Car](images/cnn-layers-car.png)

#### Convolutional Layer

The convolutional layer is produced by applying a series of many different image filters, also known as convolutional kernels, to an input image.

![CNN Layers Car](images/cnn-kernels-car.png)

In the example shown, 4 different filters produce 4 differently filtered output images. When we stack these images, we form a complete convolutional layer with a depth of 4!

#### Learning

In the code you've been working with, you've been setting the values of filter weights explicitly, but neural networks will actually learn the best filter weights as they train on a set of image data. You'll learn all about this type of neural network later in this section, but know that high-pass and low-pass filters are what define the behavior of a network like this, and you know how to code those from scratch!

In practice, you'll also find that many neural networks learn to detect the edges of images because the edges of object contain valuable information about the shape of an object.


#### Strides

One of the hyperparameters of a CNN is referred to as the _stride_ which is the amount for which the convolution window slide across the image 

![Stride CNN](images/cnn-strides-1.png)

#### Padding

// todo

#### Max Pooling Layer

We're now ready to introduce you to the second and final type of layer that we'll need to introduce before building our own convolutional neural networks.

These so-called pooling layers often take convolutional layers as input.
Recall that a convolutional layer is a stack of feature maps where we have one feature map for each filter.

A complicated dataset with many different object categories will require a large number of filters, each responsible for finding a pattern in the image.

More filters means a bigger stack, which means that the dimensionality of our convolutional layers can get quite large.
Higher dimensionality means, we'll need to use more parameters,
which can lead to over-fitting.
Thus, we need a method for reducing this dimensionality.
This is the role of pooling layers within a convolutional neural network.
We'll focus on two different types of pooling layers.
The first type is a max pooling layer,
max pooling layers will take a stack of feature maps as input.
Here, we've enlarged and visualized all three of the feature maps.
As with convolutional layers, we'll define a window size and stride.
In this case, we'll use a window size of two and a stride of two.
To construct the max pooling layer, we'll work with each feature map separately.
Let's begin with the first feature map, we start with our window in the top left corner of the image.
The value of the corresponding node in the max pooling layer is
calculated by just taking the maximum of the pixels contained in the window.
In this case, we had a one, nine, five, and four in our window,
so nine was the maximum.
If we continue this process and do it for all of our feature maps,
the output is a stack with the same number of feature maps,
but each feature map has been reduced in width and height.
In this case, the width and height are half of that of the previous convolutional layer.

#### Other kinds of pooling

Alexis mentioned one other type of pooling, and it is worth noting that some architectures choose to use average pooling, which chooses to average pixel values in a given window size. So in a 2x2 window, this operation will see 4 pixel values, and return a single, average of those four values, as output!

This kind of pooling is typically not used for image classification problems because maxpooling is better at noticing the most important details about edges and other features in an image, but you may see this used in applications for which smoothing an image is preferable.

### Convolutional Neural Networks in PyTorch

#### 1. Convolutional Layers in PyTorch

To create a convolutional layer in PyTorch, you must first import the necessary module:

```python 
import torch.nn as nn
```

Then, there is a two part process to defining a convolutional layer and defining the feedforward behavior of a model (how an input moves through the layers of a network. First you must define a Model class and fill in two functions.

**init**

You can define a convolutional layer in the ```__init__``` function of by using the following format:

```python
self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

**forward**

Then, you refer to that layer in the forward function! Here, I am passing in an input image ```x``` and applying a _ReLu_ function to the output of this layer.

```python
x = F.relu(self.conv1(x))
```

**Arguments**

You must pass the following arguments:

* ```in_channels``` - The number of inputs (in depth), 3 for an RGB image, for example.
* ```out_channels``` - The number of output channels, i.e. the number of filtered "images" a convolutional layer is made of or the number of unique, convolutional kernels that will be applied to an input.
* ```kernel_size``` - Number specifying both the height and width of the (square) convolutional kernel.

There are some additional, optional arguments that you might like to tune:

* ```stride``` - The stride of the convolution. If you don't specify anything, ```stride``` is set to ```1```.
* ```padding``` - The border of 0's around an input array. If you don't specify anything, ```padding``` is set to ```0```.

>**Note**: It is possible to represent both kernel_size and stride as either a number or a tuple

There are many other tunable arguments that you can set to change the behavior of your convolutional layers. To read more about these, we recommend perusing the official [documentation](https://pytorch.org/docs/stable/nn.html#conv2d).


#### 2. Pooling Layers

Pooling layers take in a kernel_size and a stride. Typically the same value, the is the down-sampling factor. For example, the following code will down-sample an input's x-y dimensions, by a factor of 2:

```python
self.pool = nn.MaxPool2d(2,2)
```

**forward**

Here, we see that poling layer being applied in the forward function.

```python
x = F.relu(self.conv1(x))
x = self.pool(x)
```










