
# Let's start by loading the necessary modules.

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras as ks

print(tf.__version__)



# Now, load the Fashion-MNIST data set.

(training_images, training_labels), (test_images, test_labels) = ks.datasets.fashion_mnist.load_data()



# Create class_names list object for mapping labels to names.

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# Shape of Training and Test Set

print('Training Images Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Images Dataset Labels: {}'.format(len(training_labels)))
print('Test Images Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Images Dataset Labels: {}'.format(len(test_labels)))

"""**Data Preprocessing**

As the pixel values range from 0 to 255, we have to scale these values to a range of 0 to 1 before feeding them to the model. We can scale these values (both for training and test datasets) by dividing the values by 255:
"""

training_images = training_images / 255.0

test_images = test_images / 255.0

"""**Model Building**

We will be using the keras implementation to build the different layers of a NN. We will keep it simple by having only 1 hidden layer.
"""

input_data_shape = (28, 28)
hidden_activation_function = 'relu'
output_activation_function = 'softmax'

nn_model = ks.models.Sequential()
nn_model.add(ks.layers.Flatten(input_shape=input_data_shape, name='Input_layer'))
nn_model.add(ks.layers.Dense(32, activation=hidden_activation_function, name='Hidden_layer'))
nn_model.add(ks.layers.Dense(10, activation=output_activation_function, name='Output_layer'))

nn_model.summary()

"""Now, we will use an optimization function with the help of compile method. An Adam optimizer with objective function as sparse_categorical_crossentropy which optimzes for the accuracy metric can be built as follows:"""

optimizer = 'adam'
loss_function = 'sparse_categorical_crossentropy'
metric = ['accuracy']
nn_model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)

nn_model.fit(training_images, training_labels, epochs=10)

"""**Model Evaluation**

1. Training Evaluation
"""

training_loss, training_accuracy = nn_model.evaluate(training_images, training_labels)

print('Training Data Accuracy {}'.format(round(float(training_accuracy),2)))

"""2. Test Evaluation"""

test_loss, test_accuracy = nn_model.evaluate(test_images, test_labels)

print('Test Data Accuracy {}'.format(round(float(test_accuracy),2)))