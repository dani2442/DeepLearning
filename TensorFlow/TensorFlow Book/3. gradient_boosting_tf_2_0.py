
# 1. Import the required Modules

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

print(tf.__version__)



# 2. Load anc configure the Iris Dataset

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
train = train[train.Species >= 1]
train['Species'] = train['Species'].replace([1,2], [0,1])
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
test = test[test.Species >= 1]
test['Species'] = test['Species'].replace([1,2], [0,1])

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

iris_dataset = pd.concat([train, test], axis=0)
iris_dataset.describe()



# 3. Checking the relation between the variables using Pairplot and Correlation Graph

sb.pairplot(iris_dataset, diag_king="kde")

correlation_data = iris_dataset.corr()
correlation_data.style.background_gradient(cmap='coolwarm', axis=None)



# 4. Descriptive Statistics - Central Tendency and Dispersion

stats = iris_dataset.describe()
iris_stats = stats.transpose()
iris_stats



# 5. Select the required columns

X_data = iris_dataset[[i for i in iris_dataset.columns if i not in ['Species']]]
Y_data = iris_dataset[['Species']]



# 6. Train Test Split

train_features , test_features ,train_labels, test_labels = train_test_split(X_data , Y_data , test_size=0.3)

print('Training Features Rows: ', train_features.shape[0])
print('Test Features Rows: ', test_features.shape[0])
print('Training Features Columns: ', train_features.shape[1])
print('Test Features Columns: ', test_features.shape[1])

print('Training Label Rows: ', train_labels.shape[0])
print('Test Label Rows: ', test_labels.shape[0])
print('Training Label Columns: ', train_labels.shape[1])
print('Test Label Columns: ', test_labels.shape[1])

stats = train_features.describe()
stats = stats.transpose()
stats

stats = test_features.describe()
stats = stats.transpose()
stats



# 7. Normalize Data

def norm(x):
  stats = x.describe()
  stats = stats.transpose()
  return (x - stats['mean']) / stats['std']

normed_train_features = norm(train_features)
normed_test_features = norm(test_features)



# 8. Build the Input Pipeline for TensorFlow model

def feed_input(features_df, target_df, num_of_epochs=10, shuffle=True, batch_size=35):
  def input_feed_function():
    dataset = tf.data.Dataset.from_tensor_slices((dict(features_df), target_df))
    if shuffle:
      dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size).repeat(num_of_epochs)
    return dataset
  return input_feed_function

train_feed_input = feed_input(normed_train_features, train_labels)
train_feed_input_testing = feed_input(normed_train_features, train_labels, num_of_epochs=1, shuffle=False)
test_feed_input = feed_input(normed_test_features, test_labels, num_of_epochs=1, shuffle=False)



# 9. Model Training

feature_columns_numeric = [tf.feature_column.numeric_column(k) for k in train_features.columns]

rf_model = tf.estimator.BoostedTreesClassifier(feature_columns=feature_columns_numeric, n_batches_per_layer=1)

rf_model.train(train_feed_input)



# 10. Predictions

train_predictions = rf_model.predict(train_feed_input_testing)
test_predictions = rf_model.predict(test_feed_input)

train_predictions_series = pd.Series([p['classes'][0].decode("utf-8")   for p in train_predictions])
test_predictions_series = pd.Series([p['classes'][0].decode("utf-8")   for p in test_predictions])

train_predictions_df = pd.DataFrame(train_predictions_series, columns=['predictions'])
test_predictions_df = pd.DataFrame(test_predictions_series, columns=['predictions'])

train_labels.reset_index(drop=True, inplace=True)
train_predictions_df.reset_index(drop=True, inplace=True)

test_labels.reset_index(drop=True, inplace=True)
test_predictions_df.reset_index(drop=True, inplace=True)

train_labels_with_predictions_df = pd.concat([train_labels, train_predictions_df], axis=1)
test_labels_with_predictions_df = pd.concat([test_labels, test_predictions_df], axis=1)



# 11. Validation

def calculate_binary_class_scores(y_true, y_pred):
  acc_score = accuracy_score(y_true, y_pred.astype('int64'))
  prec_score = precision_score(y_true, y_pred.astype('int64'))
  rec_score = recall_score(y_true, y_pred.astype('int64'))
  return acc_score, prec_score, rec_score


train_accuracy_score, train_precision_score, train_recall_score = calculate_binary_class_scores(train_labels, train_predictions_series)
test_accuracy_score, test_precision_score, test_recall_score = calculate_binary_class_scores(test_labels, test_predictions_series)

print('Training Data Accuracy (%) = ', round(train_accuracy_score*100,2))
print('Training Data Precision (%) = ', round(train_precision_score*100,2))
print('Training Data Recall (%) = ', round(train_recall_score*100,2))
print('-'*50)
print('Test Data Accuracy (%) = ', round(test_accuracy_score*100,2))
print('Test Data Precision (%) = ', round(test_precision_score*100,2))
print('Test Data Recall (%) = ', round(test_recall_score*100,2))






