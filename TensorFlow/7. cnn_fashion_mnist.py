# -*- coding: utf-8 -*-
"""CNN_Fashion_MNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jGYL1QCcg6XW9L2coEs_lhHmpdqrs3-l

**Convolutional Neural Network Implementation in TensorFlow 2.0**

![Convolutional Neural Network Architecture](https://miro.medium.com/max/2510/1*vkQ0hXDaQv57sALXAJquxA.jpeg)

**About the dataset**

Let us implement a simple convolutional neural network using TensorFlow 2.0. For this, we will make use of the Fashion MNIST dataset by Zalando (MIT License) which contains 70,000 images (in grayscale) in 10 different categories. The images are 28x28 pixels of individual articles of clothing with values ranging from 0 to 255 as shown below:

![Fashion MNIST dataset](https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png)

Out of the total 70,000 images, 60,000 are used for training and remaining 10,000 for testing. The labels are integer arrays ranging from 0 to 9. The class names are not a part of the dataset and hence we need to include the below mapping while training/prediction:

Label	-> Description

0	-> T-shirt/top

1	-> Trouser

2	-> Pullover

3	-> Dress

4	-> Coat

5	-> Sandal

6	-> Shirt

7	-> Sneaker

8	-> Bag

9	-> Ankle boot
"""

# Install necessary modules

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

# Validating the TensorFlow version
print(tf.__version__)

# Create class_names list object for mapping labels to names

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the Fashion MNIST dataset

mnist_fashion = ks.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist_fashion.load_data()

"""**Data Exploration**"""

# Shape of Training and Test Set

print('Training Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Dataset Labels: {}'.format(len(training_labels)))
print('Test Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Dataset Labels: {}'.format(len(test_labels)))

"""**Data Preprocessing**

As the pixel values range from 0 to 255, we have to scale these values to a range of 0 to 1 before feeding them to the model. We can scale these values (both for training and test datasets) by dividing the values by 255:
"""

training_images = training_images / 255.0

test_images = test_images / 255.0

"""Reshaping the Training and Test dataset by reshaping the matrices into 28x28x1 array: """

training_images = training_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Shape of Training and Test Set after applying reshape

print('Training Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Dataset Labels: {}'.format(len(training_labels)))
print('Test Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Dataset Labels: {}'.format(len(test_labels)))

"""**Model Building**

We will be using the keras implementation to build the different layers of a CNN. We will keep it simple by having only 2 layers. 

**First Layer - Convolutional layer with ReLU activation function:** This layer takes the 2D array (28x28 pixels) as input. We will take 50 convolutional kernels (filters) of shape 3x3 pixels, output of whose will be passed to a ReLU activation function before it is passed to the next layer.
"""

cnn_model = ks.models.Sequential()
cnn_model.add(ks.layers.Conv2D(40, (3, 3), activation='relu', input_shape=(28, 28, 1), name='Convolutional_layer'))

"""**Second Layer - Pooling layer:** This layer takes the 50 26x26 2D arrays as input and transforms them into the same number (50) of arrays with dimensions half of that of the original (i.e. from 26x26 to 13x13 pixels)"""

cnn_model.add(ks.layers.MaxPooling2D((2, 2), name='Maxpooling_2D'))

"""**Third Layer - Fully Connected layer:** This layer takes the 50 13x13 2D arrays as input and transforms them into a 1D array of 8450 elements (50x13x13). These 8450 input elements are passed through a fully connected neural network which gives out the probability scores for each of the 10 output labels (at the output layer)"""

cnn_model.add(ks.layers.Flatten(name='Flatten'))
cnn_model.add(ks.layers.Dense(50, activation='relu', name='Hidden_layer'))
cnn_model.add(ks.layers.Dense(10, activation='softmax', name='Output_layer'))

"""We can check the details of different layers built in the CNN model by using the summary method as shown below:"""

cnn_model.summary()

"""Now, we will use an optimization function with the help of compile method. An Adam optimizer with objective function as sparse_categorical_crossentropy which optimzes for the accuracy metric can be built as follows:"""

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

"""**Model Training**"""

cnn_model.fit(training_images, training_labels, epochs=10)

"""**Model Evaluation**

1. Training Evaluation
"""

training_loss, training_accuracy = cnn_model.evaluate(training_images, training_labels)

print('Training Accuracy {}'.format(round(float(training_accuracy), 2)))

"""2. Test Evaluation"""

test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels)

print('Test Accuracy {}'.format(round(float(test_accuracy), 2)))

"""From the above evaluation, we see that we were able to achieve around 97% accuracy in Training dataset and around 91% accuracy in Test dataset just with a simple CNN architecture. This goes on to prove that CNNs are powerful algorithms for Image recognition."""