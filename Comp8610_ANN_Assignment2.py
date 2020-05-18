# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:07:54 2020

@author: Shaon
"""


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)
#losding the datset
mnist = keras.datasets.mnist
#X indiciates images and y indicates lables 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
train_images = X_train/255.0
#print(X_shape.shape)
print("Number of distict labels or classes: ", np.unique(y_test))
print("Number of train images: ",len(y_train))
print("Number of test images: ",len(y_test))
#some sample dataset images
print("Dataset Samples: ")
class_names = ['Class 0','Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i],)
    plt.xlabel(class_names[y_train[i]])
plt.show()

# Preprocess the data (these are Numpy arrays)
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

# Reserve 10,000 samples for validation
X_val = X_train[-10000:]
y_val = y_train[-10000:]
X_train = X_train[:-10000]
y_train = y_train[:-10000]

# =============================================================================
# #Question No: 1(1):
# #Try the basic minibatch SGD. It is recommended to try different initializations, different batch
# #sizes, and different learning rates, in order to get a sense about how to tune the
# #hyperparameters (batch size, and, learning rate). Remember to create and use validation
# #dataset!. it will be very useful for you to read Chapter-11 of the textbook.
# 
# =============================================================================
#Shallow neural network model using MiniBatch SGD
print("#Shallow neural network model using ADAM optimizer")
eta = 0.1
mini_batch =10000
epoch_size = 15
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),  # SGD Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose=0)

#print('\nhistory dict:', history.history)
print("# Fit model on training data using SGD Minibatch optimizer") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

#Shallow neural network model using MiniBatch SGD
eta = 0.01
mini_batch =500 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    #keras.layers.Dense(128, activation='relu'),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),  # SGD Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data using SGD Minibatch optimizer") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
#Shallow neural network model using MiniBatch SGD
eta = 0.001
mini_batch =100 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),  # SGD Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val),verbose =0)

print("# Fit model on training data using SGD Minibatch optimizer") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
# =============================================================================
#Question 1(2): 
#It is recommended to try, at least, another optimization method (SGD momentum, RMSProp,
# RMSProp momentum, AdaDelta, or Adam) and compare its performances to those of the
# basic minibatch SGD on the MNIST dataset. Which methods you want to try and how many
# you want to try and compare is up to you and up to the amount of time you have left to
# complete the assignment. Remember, this is a research course. You may want to read
# Chapter-8, which I will cover this week.
# #Shallow neural network model using RMSprop optimizer
# =============================================================================
print("#Shallow neural network model using RMSprop optimizer")
eta = 0.01
mini_batch =500 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(keras.optimizers.RMSprop(learning_rate=eta),  # ADAM Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data usint RMSprop optimizer") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
########################################################################
#Shallow neural network model using ADAM optimizer
print("#Shallow neural network model using ADAM optimizer")
eta = 0.01
mini_batch =500 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=eta),  # ADAM Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data using ADAM optimizer:") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
##############################################################################
#Deep neural network model using SGD Minibatch optimizer
print("#Deep neural network model using SGD Minibatch optimizer")
eta = 0.01
mini_batch =500 
epoch_size =30
hidden_layer_size =100
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dense(10,activation='softmax')])
model.summary()
model.compile(keras.optimizers.SGD(learning_rate=eta),  # ADAM Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data usint SGD Minibatch optimizer") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
###############################################################################
#Deep neural network model using RMSprop optimizer
print("#Deep neural network model using RMSprop optimizer")
eta = 0.01
mini_batch =500 
epoch_size =30
hidden_layer_size =100
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(keras.optimizers.RMSprop(learning_rate=eta),  # ADAM Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data usint RMSprop optimizer") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
##################################################################################
#Deep neural network model using ADAM optimizer
print("#Deep neural network model using ADAM optimizer")
eta = 0.01
mini_batch =500 
epoch_size =30
hidden_layer_size =100
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=eta),  # ADAM Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])
history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data using ADAM optimizer:") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

# =============================================================================
# Question: 2 
# Consider the L2 regularized multiclass logistic regression. That is, add to the logistic
# regression loss a regularization term that represents L2 norm of the parameters. 
#More precisely, the regularization term is λi(∥wi∥2 + ∥bi∥2)
# where {wi, bi} are all the parameters in the logistic regression, and λ ∈ R is the regularization hyper-
# parameter. Typically, λ is about C/n where n is the number of data points and C is some constant in
# [0.01,100] (need to tune C). Run the regularized multiclass logistic regression on MNIST, using the 
# basic minibatch SGD, and compare its results to those of the basic minibatch SGD with non-
# regularized loss, in Question #1. 
# =============================================================================
print("Logistic Regression using L2 regularization")
from sklearn.linear_model import LogisticRegression 
constant = 50.0
norm = 'l2'
epoch_size =30
classifier = LogisticRegression(penalty=norm, dual=False, tol=0.0001, C=constant, 
                                max_iter =epoch_size, random_state=42)
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = classifier.score(X_test,y_test)*100
print("Parameter: Norm=",norm, ',C=',constant, ',epochs=',epoch_size)
print('Accuracy = ',score, '%')
print(classification_report(y_test, y_pred))

#Applying Grid Search technique to get best value of C
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1,10,30,50,100]}]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, cv=2, scoring='accuracy')
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters= grid_search.best_params_
print("After applying K-fold Cross validation and Grid Search Technique for C=[1,10,30,50,100]"
      +"\nthe best parameters are found as follows:")
print(best_parameters)
#output:
#After applying K-fold Cross validation and Grid Search Technique
#the best parameters are found as follows:
#{'C': 1}

print("Logistic Regression using L2 regularization")
from sklearn.linear_model import LogisticRegression 
constant = 1.0
norm = 'l2'
epoch_size =30
classifier = LogisticRegression(penalty=norm, dual=False, tol=0.0001, C=constant, 
                                max_iter =epoch_size, random_state=42)
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = classifier.score(X_test,y_test)*100
print("Parameter: Norm=",norm, 'C=',constant, 'epochs=',epoch_size)
print('Accuracy = ',score, '%')
print(classification_report(y_test, y_pred))
#######################################################
print("Logistic Regression using L1 regularization")
from sklearn.linear_model import LogisticRegression 
constant = 1.0
norm = 'l1'
epoch_size =30
classifier = LogisticRegression(penalty=norm, dual=False, tol=0.0001, C=constant, 
                                max_iter =epoch_size, random_state=42)
classifier = classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
score = classifier.score(X_test,y_test)*100
print("Parameter: Norm=",norm, 'C=',constant, 'epochs=',epoch_size)
print('Accuracy = ',score, '%')
print(classification_report(y_test, y_pred))
#################################################################################
print("#Shallow neural network model using SGD Minibatch optimizer with L2 regularization")
eta = 0.01
mini_batch =500 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(10,activation='softmax',
                                       activity_regularizer=keras.regularizers.l2(0.01))])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),  # SGD Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data using SGD Minibatch optimizer with L2 regularization") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
#############################################################################################
print("#Shallow neural network model using RMSprop  optimizer with L2 regularization")
eta = 0.01
mini_batch =500 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(10,activation='softmax',
                                       activity_regularizer=keras.regularizers.l2(0.01))])

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=eta),  # SGD Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data using RMSprop optimizer with L2 regularization") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
#########################################################################################
print("#Shallow neural network model using ADAM optimizer with L2 regularization")
eta = 0.01
mini_batch =500 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(10,activation='softmax',
                                       activity_regularizer=keras.regularizers.l2(0.01))])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=eta),  # SGD Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data using ADAM optimizer with L2 regularization") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
#Then compare the best value that we got after parameter tuning in question 1(1) in SGD minibatch 
#
#
#####
#

# =============================================================================
# Question: 3 
# Going above and beyond Question-1 and Question-2, investigate the basic minibatch SGD with, 
# at least, another regularization method discussed in class (L1, data augmentation, noise
# robustness, early stopping, sparse representation, bagging, or dropout). Currently, L2 norm, early
# stopping, and dropout are the most frequently used regularization methods. You may need to read
# Chapter-7, which I have started to cover in class. You may even try CNN if time allows you.
# =============================================================================
print("#Shallow neural network model using SGD Minibatch optimizer with L1 regularization")
eta = 0.01
mini_batch =500 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(10,activation='softmax',
                                       activity_regularizer=keras.regularizers.l1(0.01))])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),  # SGD Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data using SGD Minibatch optimizer with L2 regularization") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
##########################################################################
print("#Shallow neural network model using SGD Minibatch optimizer with Dropout regularization")
eta = 0.01
mini_batch =500 
epoch_size =30
model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(10,activation='softmax',
                                       activity_regularizer=keras.regularizers.l1(0.01))])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=eta),  # SGD Optimizer
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), verbose =0)

print("# Fit model on training data using SGD Minibatch optimizer with Drop regularization") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['sparse_categorical_accuracy'])
axes[0].plot(history.history['val_sparse_categorical_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
################################################################################
print("#Deep neural network model using SGD Minibatch optimizer with Dropout Regularizer")
eta = 0.01
mini_batch =500 
epoch_size =300
hidden_layer_size =100

early_stopping = tf.keras.callbacks.EarlyStopping(patience =2)

model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dropout(0.2),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dropout(0.1),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam',  
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), 
                    callbacks = [early_stopping], verbose =2)
print("# Fit model on training data using SGD Minibatch optimizer with Dropout Regularizer") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

##########################################################################
print("#Deep neural network model using SGD Minibatch optimizer with Early Stopping")
eta = 0.01
mini_batch =500 
epoch_size =300
hidden_layer_size =100

early_stopping = tf.keras.callbacks.EarlyStopping(patience =2)

model = keras.Sequential([keras.Input(shape=(784,)),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dense(hidden_layer_size, activation='relu'),
                    keras.layers.Dense(10,activation='softmax')])

model.compile(optimizer='adam',  
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=mini_batch,
                    epochs=epoch_size,
                    validation_data=(X_val, y_val), 
                    callbacks = [early_stopping], verbose =2)
print("# Fit model on training data using SGD Minibatch optimizer with Early Stopping") 
print( "Learning Rate=",eta,", Mini batch size=",mini_batch,", Epochs=",epoch_size)
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
#########################################################################
#Applying CNN to mnist dataset with Dropout Regularizatoin
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 500
num_classes = 10
epochs_size = 30
img_rows =28
img_cols = 28
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# Reserve 10,000 samples for validation
X_val = X_train[-10000:]
y_val = y_train[-10000:]
X_train = X_train[:-10000]
y_train = y_train[:-10000]
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_val.shape[0], 'validation samples')

#convert class vectors to binary class matrices
#we can avoid this step by using sparse_categorical_crossentropy loss
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
#y_val = keras.utils.to_categorical(y_val, num_classes)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print("Model Information: ")
model.summary()
import time
starting_time = time.time()
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs_size,
          verbose=1,
          validation_data=(X_val, y_val))
ending_time = time.time()
total_time = ending_time-starting_time
print('Total Time Taken: {0:.2f} seconds'.format(total_time))
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))

##############################################################################
#Applying CNN to MNIST Dataset with Dropout and Early Stopping Regularization
early_stopping = tf.keras.callbacks.EarlyStopping(patience =1)
starting_time = time.time()
history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs_size,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks = [early_stopping])
ending_time = time.time()
total_time = ending_time-starting_time
print('Total Time Taken: {0:.2f} seconds'.format(total_time))
#visulatizing model learning accuracy and loss on training and validation data
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
# Plot training & validation accuracy values
axes[0].plot(history.history['accuracy'])
axes[0].plot(history.history['val_accuracy'])
axes[0].set_title('Model accuracy')
axes[0].set_ylabel('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend(['Train', 'Validation'], loc='upper left')
# Plot training & validation loss values
axes[1].plot(history.history['loss'])
axes[1].plot(history.history['val_loss'])
axes[1].set_title('Model loss')
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()
# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('Test loss: {0:.2f}, Test Accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100)) 
from sklearn.metrics import classification_report
y_pred = model.predict(X_test, batch_size=mini_batch, verbose=0)
y_pred_bool = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred_bool))
