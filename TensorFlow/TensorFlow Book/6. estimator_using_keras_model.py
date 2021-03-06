# -*- coding: utf-8 -*-
# Estimator_Using_Keras_Model.ipynb

# Automatically generated by Colaboratory.

# Original file is located at
#     https://colab.research.google.com/drive/1ZDce94MmLZC65NRNkOxcIh7tTbiyIoAb


# Import the required Modules

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_datasets as tf_ds

print(tf.__version__)

def data_input():
  train_test_split = tf_ds.Split.TRAIN
  iris_dataset = tf_ds.load('iris', split=train_test_split, as_supervised=True)
  iris_dataset = iris_dataset.map(lambda features, labels: ({'dense_input':features}, labels))
  iris_dataset = iris_dataset.batch(32).repeat()
  return iris_dataset

activation_function = 'relu'
input_shape = (4,)
dropout = 0.2
output_activation_function = 'sigmoid'

keras_model = ks.models.Sequential([ks.layers.Dense(16, activation=activation_function, input_shape=input_shape), ks.layers.Dropout(dropout), ks.layers.Dense(1, activation=output_activation_function)])

loss_function = 'categorical_crossentropy'
optimizer = 'adam'

keras_model.compile(loss=loss_function, optimizer=optimizer)
keras_model.summary()

model_path = "/keras_estimator/"
estimator_keras_model = ks.estimator.model_to_estimator(keras_model=keras_model, model_dir=model_path)

estimator_keras_model.train(input_fn=data_input, steps=25)
evaluation_result = estimator_keras_model.evaluate(input_fn=data_input, steps=10)
print('Final evaluation result: {}'.format(evaluation_result))
