# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:27:35 2020

@author: Shaon
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
"""
Question 1(a): 
Create a vector, x,containing 5000 observations drawn from a Gaussian distribution N(0, 1) 
[ie, a normal distribution with mean 0 and variance 1].
"""
#intializing mean(mu) and starndard deviation(sigma) for x
mu_x, sigma_x = 0, 1
#variance = sigma^2 = 1 
#creating vector x according to the given criteria. 
x = np.random.normal(mu_x,sigma_x,5000)

"""
Question 1(b): 
Create a vector, eps, containing 5000 observation drawn from a N(0, 0.25) distribution; 
ie, a normal distribution with mean 0 and variance 0.25.
"""
#intializing mean(mu) and starndard deviation(sigma) for eps
mu_eps, sigma_eps = 0, 0.5
#variance = sigma^2 = 0.5^2 = 0.25 
#creating vector eps according to the given criteria. 
eps = np.random.normal(mu_eps,sigma_eps,5000)
"""
Question 1(c):
Using vectors x and eps, generate a vector y according to the model: y = -1 + 0.5x – 2x2 + 0.3x3 + eps.
Your 5000 data-points (x, y) are generated upon completion of this Part-c. 
Note that the true function is a cubic function with true weight vector being wtrue = (-1, +0.5, -2, +0.3).
"""
y = -1 + 0.5*x - 2*x**2 + 0.3*x**3 + eps

print("\nVisualizing the Synthetic Dataset")
plt.style.use("ggplot")
plt.scatter(x,y,color='red',edgecolors="green")
plt.title("Synthetic Dataset")
plt.xlabel("X", fontsize=20)
plt.ylabel("y",rotation = 0, fontsize = 20)
plt.show()
"""
Question 1(d):	
Implement the Adaline neuron learning algorithm using (i) batch gradient descent and (ii) stochastic gradient descent,
and test and compare them on linear regression over your synthetic data-points. 
You need not to perform a cross-validation scheme here; only use the whole data set as your training set. 
"""

X = np.asanyarray(x).reshape(-1,1) #x need to be converted into matrix without changing the array values to fit the model
eta1 = 0.0001
eta2 = 0.1

from mlxtend.regressor import LinearRegression
from sklearn import metrics
ada1_bgd = LinearRegression (method='sgd',
                          eta=eta1, 
                          epochs=20, 
                          random_seed=0, 
                          minibatches=1) #for adalline bgd
ada1_bgd.fit(X, y)
y_pred = ada1_bgd.predict(X)
mse1 = metrics.mean_squared_error(y_pred,y)
ada2_bgd = LinearRegression(method='sgd',
                          eta=eta2, 
                          epochs=20, 
                          random_seed=0, 
                          minibatches=1) #for adaline bgd
ada2_bgd.fit(X,y)
y_pred = ada2_bgd.predict(X)
mse2 = metrics.mean_squared_error(y_pred,y)
print("Adaline Batch Gradient Descent Regression Algorithm")
print("-----------------------------------------------------")
print("\tLearning Rate: ",eta1, "\t\t\tLearning Rate: ",eta2)
print('\tIntercept: %.2f' % ada1_bgd.w_, end='')
print('\t\t\t\tIntercept: %.2f' % ada2_bgd.w_)
print('\tSlope: %.2f' % ada1_bgd.b_, end='')
print('\t\t\t\tSlope: %.2f' % ada2_bgd.b_,)
print('\tMSE: ', mse1, end='')
print('\t\t\tMSE: ', mse2)
xp1 = np.linspace(x.min(),x.max(),5000)
xp2 = np.linspace(x.min(),x.max(),5000) 
#Visulizing BGD Regression and Cost Plot
plt.style.use("ggplot")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter(X,y,color='red',edgecolors='green',label='Synthetic Data Point')
ax[0].plot(xp1,ada1_bgd.predict(xp1.reshape(-1,1)),color='blue',label="Regression Line")
ax[0].set_title("AdalineBGD X vs Y (LR:"+str(eta1)+")")
ax[0].set_xlabel("X", fontsize=20)
ax[0].set_ylabel("y",rotation = 0, fontsize = 20)
ax[0].legend()
ax[1].scatter(X,y,color='red',edgecolors="green",label="Synthetic Data Point")
ax[1].plot(xp2,ada2_bgd.predict(xp2.reshape(-1,1)),color='blue',label="Regression Line")
ax[1].set_title("AdalineBGD X vs Y (LR:"+str(eta2)+")")
ax[1].set_xlabel("X", fontsize=20)
ax[1].set_ylabel("y",rotation = 0, fontsize = 20)
ax[1].legend()
plt.tight_layout()
plt.show()
fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax1[0].plot(range(1, ada1_bgd.epochs + 1), np.log10(ada1_bgd.cost_), marker='d')
ax1[0].set_xlabel('Epochs')
ax1[0].set_ylabel('Cost: Log (Sum Squared Error)')
ax1[0].set_title('AdalineBGD - Learning rate '+str(eta1))
ax1[1].plot(range(1, ada2_bgd.epochs + 1), np.log10(ada2_bgd.cost_), marker='d')
ax1[1].set_xlabel('Epochs')
ax1[1].set_ylabel('Cost: Log (Sum Squared Error)')
ax1[1].set_title('AdalineBGD - Learning rate'+str(eta2))
plt.tight_layout()
plt.show()

print("\t\tAdaline Stochastic Gradient Descent Regression Algorithm")
print("\t\t--------------------------------------------------------")
ada1_sgd = LinearRegression(method='sgd',
                          eta=eta1, 
                          epochs=20, 
                          random_seed=0, 
                          minibatches=len(y)) #for adalline sgd
ada1_sgd.fit(X, y)
y_pred = ada1_sgd.predict(X)
mse1 = metrics.mean_squared_error(y_pred,y)

ada2_sgd = LinearRegression(method='sgd',
                          eta=eta2, 
                          epochs=20, 
                          random_seed=0, 
                          minibatches=len(y)) #for adaline sgd
ada2_sgd.fit(X, y)
y_pred = ada2_sgd.predict(X)
mse2 = metrics.mean_squared_error(y_pred,y)

print("\tLearning Rate: ",eta1, "\t\t\tLearning Rate: ",eta2)
print('\tIntercept: %.2f' % ada1_sgd.w_, end='')
print('\t\t\t\tIntercept: %.2f' % ada2_sgd.w_)
print('\tSlope: %.2f' % ada1_sgd.b_, end='')
print('\t\t\t\tSlope: %.2f' % ada2_sgd.b_,)
print('\tMSE: ', mse1, end='')
print('\t\t\tMSE: ', mse2)

xp1 = np.linspace(x.min(),x.max(),5000)
xp2 = np.linspace(x.min(),x.max(),5000) 
#Visulizing SGD Regression and Cost Plot
plt.style.use("ggplot")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter(X,y,color='red',edgecolors="green",label="Synthetic Data Point")
ax[0].plot(xp1,ada1_sgd.predict(xp1.reshape(-1,1)),color='blue',label="Regression Line")
ax[0].set_title("AdalineSGD X vs Y (LR:"+str(eta1)+")")
ax[0].set_xlabel("X", fontsize=20)
ax[0].set_ylabel("y",rotation = 0, fontsize = 20)
ax[0].legend()
ax[1].scatter(X,y,color='red',edgecolors="green",label="Synthetic Data Point")
ax[1].plot(xp2,ada2_sgd.predict(xp2.reshape(-1,1)),color='blue',label="Regression Line")
ax[1].set_title("AdalineSGD X vs Y (LR:"+str(eta2)+")")
ax[1].set_xlabel("X", fontsize=20)
ax[1].set_ylabel("y",rotation = 0, fontsize = 20)
ax[1].legend()
plt.tight_layout()

plt.show()
fig, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ax1[0].plot(range(1, ada1_sgd.epochs + 1), np.log10(ada1_sgd.cost_), marker='d')
ax1[0].set_xlabel('Epochs')
ax1[0].set_ylabel('Cost: Log (Sum Squared Error)')
ax1[0].set_title('AdalineSGD - Learning rate '+str(eta1))
ax1[1].plot(range(1, ada2_sgd.epochs + 1), np.log10(ada2_sgd.cost_), marker='d')
ax1[1].set_xlabel('Epochs')
ax1[1].set_ylabel('Cost: Log (Sum Squared Error)')
ax1[1].set_title('AdalineSGD - Learning rate'+str(eta2))
plt.tight_layout()
plt.show()

print("\t\t\t\tLinear Regression Algorithm")
print("\t\t\t\t----------------------------")
from sklearn.linear_model import LinearRegression 
lr_model = LinearRegression()
X = np.asanyarray(x).reshape(-1,1) #x need to be converted into matrix without changing the array values to fit the model
lr_model.fit(X,y)
y_pred = lr_model.predict(X)
mse = metrics.mean_squared_error(y,y_pred)
print("MSE: ", mse)
#visualizing the training set result for Linear Regression
plt.style.use("ggplot")
plt.scatter(X,y,color='red',edgecolors="green",label="Synthetic Data Point")
plt.plot(X,lr_model.predict(X),color='blue',label="Regression Line")
plt.title("Linear Regression Plot (X vs y)")
plt.xlabel("X", fontsize=20)
plt.ylabel("y",rotation = 0, fontsize = 20)
plt.legend()
plt.show()
"""
Questino 1(e): Repeat Part-d, but with a Sigmoid neuron of your choice (logistic or tanh).
"""

import tensorflow as tf
X = x.reshape(-1,1)
Y = y.reshape(-1,1)

model = tf.keras.Sequential([
              tf.keras.layers.Dense(1, input_shape=(1,),activation='tanh')

        ])
model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss='mean_squared_error', metrices=['mean_squared_error'])
model.fit(X,Y, epochs = 30, batch_size =512)
y_pred = model.predict(X)
print("\t\t Single layer Sigmoid Nueron using tanh activation function ")
print("\t\t\t----------------------------------------------------------")
mse = metrics.mean_squared_error(y,y_pred)
print("Result: \n---------------")
print("MSE: ", mse)
xp = np.linspace(x.min(), x.max())
plt.style.use("ggplot")
plt.scatter(X,y,color='red', edgecolors='green', label='Synthetic Data Points')
plt.plot(xp,model.predict(xp.reshape(-1)),color='blue', label='Regression Line')
plt.title("tanh Activation Plot (X vs y)")
plt.xlabel("X", fontsize=20)
plt.ylabel("y",rotation = 0, fontsize = 20)
plt.legend()
plt.show()

"""
Questino 1(f): 
Repeat Part-d and Part-e with cross-validation method of your own choice (LOOCV or 10-fold-cv) to find the best degree d. 
You must first randomly create a test set of size between 20% and 30% drawn from your original full data set. 
If this is correctly done, your methods should not only find d = 3 to be the degree, 
but they should also find the best weight vector, wbest, to be as close as possible to wtrue = (-1, +0.5, -2, +0.3).
"""

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
lm = LinearRegression()

crossvalidation = KFold(n_splits=10, random_state=0, shuffle=False)
min_mse = 23432142134.2343
min_degree =1
for i in range(1,11):
    poly = PolynomialFeatures(degree=i)
    X_poly = poly.fit_transform(X_train)
    poly.fit(X_poly, y_train)
    model = lm.fit(X_poly, y_train)
    scores = cross_val_score(model, X_poly, y_train, scoring="neg_mean_squared_error", cv=crossvalidation, n_jobs=1)
    mse = np.mean(np.abs(scores))
    print("Degree: "+str(i)+", \nPolynomial MSE: " + str(mse) + ", STD: " + str(np.std(scores)))
    if(min_mse>mse):
        min_mse = mse
        min_degree = i
    
    x_train_arr = np.asarray(X_train).reshape(-1)
    y_train_arr = np.asarray(y_train).reshape(-1)
    x_test_arr = np.asarray(X_test).reshape(-1)
    y_test_arr = np.asarray(y_test).reshape(-1)
    weights = np.polyfit(x_train_arr,y_train_arr,i)
    #generating model with the given weights
    model = np.poly1d(weights)
    xp_train = np.linspace(x_train_arr.min(), x_train_arr.max())
    xp_test = np.linspace(x_test_arr.min(), x_test_arr.max(), 70)
    pred_plot_train = model(xp_train)
    pred_plot_test = model(xp_test)
    
    print("Weights:")
    for j in range(0,len(weights)):
        print("w"+str(j)+" = "+str(weights[j]))
    # Visualising the Polynomial Regression results
    plt.style.use("ggplot")
    fig, ax = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
    ax[0].scatter (X_train,y_train,color='red', edgecolors='green', label='Synthetic Data Point')
    ax[0].plot(xp_train,pred_plot_train,color='blue',label='Regression Line')
    ax[0].set_title("Plot (Training Set), Degree="+str(i))
    ax[0].set_xlabel("X_train", fontsize=20)
    ax[0].set_ylabel("y_train", fontsize = 20)
    ax[0].legend()
    ax[1].scatter(X_test,y_test,color='red', edgecolors='green', label='Sysnthetic Data Point')
    ax[1].plot(xp_test, pred_plot_test,color='blue', label='Regression Line')
    ax[1].set_title("Plot (Testing Set), Degree="+str(i))
    ax[1].set_xlabel("X_test", fontsize=20)
    ax[1].set_ylabel("y_test", fontsize = 20)
    ax[1].legend()
    plt.tight_layout()
    plt.show()

print("From the above MSE values for various degree we have found \nthe best and least mean square error as folows:   ")
print("Best Degree: "+str(min_degree)+", MSE: "+str(min_mse))

#Again Implementing Neuron Using tanh activatin function using training and test set
model = tf.keras.Sequential([
              tf.keras.layers.Dense(1, input_shape=(1,),activation='tanh')

        ])
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mean_squared_error', metrices=['mean_squared_error'])
model.fit(X_train,y_train, epochs = 30, batch_size =512)

print("\tSingle layer Sigmoid Nueron using tanh activation function using train and test set ")
print("\t--------------------------------------------------------------------------------------")
y_pred = model.predict(X_test)
mse = metrics.mean_squared_error(y_test,y_pred)
print("Testset Result: \n---------------")
print("MSE: ", mse)

fig, ax = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter (X_train, y_train, color='red', edgecolors='green', label='Synthetic Data Points')
ax[0].plot(xp_train,model.predict(xp_train.reshape(-1)),color='blue', label='Regression Line')
ax[0].set_title("tanh Activation Plot (Training Set)")
ax[0].set_xlabel("X_train", fontsize=20)
ax[0].set_ylabel("y_train", fontsize = 20)
ax[0].legend()
ax[1].scatter(X_test,y_test,color='red', edgecolors='green', label='Synthetic Data Points')
ax[1].plot(xp_test,model.predict(xp_test.reshape(-1)),color='blue',label='Regression Line')
ax[1].set_title("tanh Activation Plot (Testing Set)")
ax[1].set_xlabel("X_test", fontsize=20)
ax[1].set_ylabel("y_test", fontsize = 20)
ax[1].legend()
plt.tight_layout()
plt.show()

"""
Questino 1(g):
Use your creativity and do whatever experiments you want to test, and then tell me whatever story your experiments told you.  

"""
X = x.reshape(-1,1)
Y = y.reshape(-1,1)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,)),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(1)
        ])
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mean_squared_error', metrices=['mean_squared_error'])
model.fit(X_train,y_train, epochs = 30, batch_size =512)

print("\tTwo layered Sigmoid Nueron using tanh activation function using train and test set ")
print("\t--------------------------------------------------------------------------------------")
y_pred = model.predict(X_test)
mse = metrics.mean_squared_error(y_test,y_pred)
print("Testset Result: \n---------------")
print("MSE: ", mse)

fig, ax = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter (X_train, y_train, color='red', edgecolors='green', label='Synthetic Data Points')
ax[0].plot(xp_train,model.predict(xp_train.reshape(-1)),color='blue', label='Regression Line')
ax[0].set_title("tanh Activation Plot (Training Set)")
ax[0].set_xlabel("X_train", fontsize=20)
ax[0].set_ylabel("y_train", fontsize = 20)
ax[0].legend()
ax[1].scatter(X_test,y_test,color='red', edgecolors='green', label='Synthetic Data Points')
ax[1].plot(xp_test,model.predict(xp_test.reshape(-1)),color='blue',label='Regression Line')
ax[1].set_title("tanh Activation Plot (Testing Set)")
ax[1].set_xlabel("X_test", fontsize=20)
ax[1].set_ylabel("y_test", fontsize = 20)
ax[1].legend()
plt.tight_layout()
plt.show()

#Let's add one more layer
import tensorflow as tf
X = x.reshape(-1,1)
Y = y.reshape(-1,1)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,)),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(100, activation='tanh'),
        tf.keras.layers.Dense(200, activation='tanh'),        
        tf.keras.layers.Dense(1)
        ])
model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mean_squared_error', metrices=['mean_squared_error'])
model.fit(X_train,y_train, epochs = 30, batch_size =512)

print("\tthree layered Sigmoid Nueron using tanh activation function using train and test set ")
print("\t--------------------------------------------------------------------------------------")
y_pred = model.predict(X_test)
mse = metrics.mean_squared_error(y_test,y_pred)
print("Testset Result: \n---------------")
print("MSE: ", mse)

fig, ax = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
ax[0].scatter (X_train, y_train, color='red', edgecolors='green', label='Synthetic Data Points')
ax[0].plot(xp_train,model.predict(xp_train.reshape(-1)),color='blue', label='Regression Line')
ax[0].set_title("tanh Activation Plot (Training Set)")
ax[0].set_xlabel("X_train", fontsize=20)
ax[0].set_ylabel("y_train", fontsize = 20)
ax[0].legend()
ax[1].scatter(X_test,y_test,color='red', edgecolors='green', label='Synthetic Data Points')
ax[1].plot(xp_test,model.predict(xp_test.reshape(-1)),color='blue',label='Regression Line')
ax[1].set_title("tanh Activation Plot (Testing Set)")
ax[1].set_xlabel("X_test", fontsize=20)
ax[1].set_ylabel("y_test", fontsize = 20)
ax[1].legend()
plt.tight_layout()
plt.show()

"""
Questino 2(a):
a.	Randomly create 2500 data-points (x, y)’s of class -1 to lie one side of the function f above 
and 2500 data-points (x, y)’s of class +1 to lie on the other side of the function. 
Indeed, here, you are not required to create your data using the function f above; 
You can use any function you want, as long as it is a simple linearly separable function of your 
choice to be used to separate 5000 data points into two classes 
"""
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5000, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=2,class_sep=1.5,flip_y=0, random_state=0,shuffle=False)

for i, j in enumerate(np.asarray(y)):
    if j==0:
        y[i] = -1
 
unique_elements, counts_elements = np.unique(y, return_counts=True)
print("Frequency of unique class of the array:")
print(np.asarray((unique_elements, counts_elements)))
print("\nVisualizing the synthetic dataset of Class 1 and Class -1: ")
plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], 'g^', label='Class: -1')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'o', label="Class: 1")
plt.title("Dataset")
plt.xlabel("X1")
plt.xlabel("X2")
plt.legend()
plt.margins()
plt.show()     
"""
Question 2(b):
Implement the Perceptron learning algorithm and run it on your synthetic data set. 
Obtain the best Perceptron model via any cross-validation method of your choice. Use your creativity 
to tell me anything about your Perceptron: for example, how does the performance (speed and accuracy) 
vary when changing the size of the training set from 50% to 80%? Also, how does Perceptron compare
with the Adaline and Sigmoid neurons on the same training sets? 
[note here that you are using Adaline and Sigmoid on a classification problem, not a regression problem].
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Perceptron to the Training set
from sklearn.linear_model import Perceptron 
classifier = Perceptron (random_state=42,alpha =0.01,eta0=0.2,max_iter=100)
classifier.fit(X_train, y_train)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv =10 )
print("Cross validation accuracy and variance of Perceptron on the Traing Set :") 
acc = accuracies.mean()*100
print('Mean Accuracy: %.2f'% acc,'%' )
print("STD: ", accuracies.std())

print("Test set result: ")
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn import metrics
cm = metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n ",cm)
print( "{0}".format(metrics.classification_report(y_test,y_pred)))
accuracy_test = metrics.accuracy_score(y_test,y_pred)*100
print('Accuracy:%.2f' % accuracy_test,"%")

# Visualising the Training set and Test set results
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
fig1 = plot_decision_regions(X_train, y_train, clf=classifier, ax=axes[0], legend=0)
fig2 = plot_decision_regions(X_test, y_test, clf=classifier, ax=axes[1], legend=0)

axes[0].set_title('Perceptron (Training set)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[1].set_title('Perceptron (Test set)')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles, 
          ['class -1', 'class 1'])
fig2.legend(handles, 
          ['class -1', 'class 1'])

plt.tight_layout()
plt.show()



#Fitting Adaline (SGD) to the training set
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(random_state=42,max_iter=200)
classifier.fit(X_train, y_train)

print("Cross validation accuracy and variaance of Adaline on the Traing Set:") 
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv =10 )
acc = accuracies.mean()*100
print('Mean Accuracy: %.2f'% acc,'%' )
print("STD: ", accuracies.std())

print("Test set result: ")
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n ",cm)
print( "{0}".format(metrics.classification_report(y_test,y_pred)))
accuracy_test = metrics.accuracy_score(y_test,y_pred)*100
print('Accuracy:%.2f' % accuracy_test,"%")

# Visualising the Training set and Test set results
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
fig1 = plot_decision_regions(X_train, y_train, clf=classifier, ax=axes[0], legend=0)
fig2 = plot_decision_regions(X_test, y_test, clf=classifier, ax=axes[1], legend=0)

axes[0].set_title('Adaline SGD (Training set)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[1].set_title('Adaline SGD (Test set)')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles, 
          ['class -1', 'class 1'])
fig2.legend(handles, 
          ['class -1', 'class 1'])

plt.tight_layout()
plt.show()

# Fitting Sigmoid model to the Training set
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
def create_neural_network():
    model = Sequential()
    model.add(Dense(units=1,input_dim=2, kernel_initializer='uniform', activation='sigmoid'))
    model.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
    return model

neuralNet = KerasClassifier(build_fn=create_neural_network, epochs=20, batch_size=500)
neuralNet.fit(X_train,y_train, epochs = 20, batch_size=500)
accuracies_neural= cross_val_score(estimator=neuralNet, X=X_train, y=y_train, cv =10 )
print("Cross validation average accuracy and std of Sigmoid Neuron on the Traing Set:") 
acc = accuracies_neural.mean()*100
print('Mean Accuracy: %.2f'% acc,'%' )
print("STD: ", accuracies_neural.std())

print("Test set result: ")

# Predicting the Test set results
y_pred_sig = neuralNet.predict(X_test)
y_pred_sig = (y_pred_sig>0.5)
#Confusin Matrix
from sklearn import metrics
y_pred_sig = neuralNet.predict(X_test)
for i in range(0,len(y_pred_sig)):
    if(y_pred_sig[i]>0.5):
        y_pred_sig[i]=1
    else:
        y_pred_sig[i]=-1
    
cm = metrics.confusion_matrix(y_test,y_pred_sig)
print("Confusion Matrix:\n ",cm)
print( "{0}".format(metrics.classification_report(y_test,y_pred_sig)))
accuracy_test = metrics.accuracy_score(y_test,y_pred_sig)*100
print('Accuracy:%.2f' % accuracy_test,"%")

# Visualising the Training set results
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
fig1 = plot_decision_regions(X_train, y_train, clf=neuralNet, ax=axes[0], legend=0)
fig2 = plot_decision_regions(X_test, y_test, clf=neuralNet, ax=axes[1], legend=0)

axes[0].set_title('Sigmoid (Training set)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[1].set_title('Sigmoid (Test set)')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles, 
          ['class -1', 'class 1'])
fig2.legend(handles, 
          ['class -1', 'class 1'])

plt.tight_layout()
plt.show()


#implementing same algorithms after with making the traing set 50%
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
#Repeat the 2(b) again and so on... 



#after adding hidden layers to the same sigmoid neuron 
def create_neural_network():
    model = Sequential()
    model.add(Dense(units=20, input_dim=2, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
    return model

neuralNet = KerasClassifier(build_fn=create_neural_network, epochs=20, batch_size=500)
neuralNet.fit(X_train,y_train, epochs = 20, batch_size=500)
accuracies_neural= cross_val_score(estimator=neuralNet, X=X_train, y=y_train, cv =10 )
print("Cross validation average accuracy and std of Sigmoid Neuron on the Traing Set:") 
acc = accuracies_neural.mean()*100
print('Mean Accuracy: %.2f'% acc,'%' )
print("STD: ", accuracies_neural.std())

print("Test set result: ")
# Predicting the Test set results
from sklearn import metrics
y_pred_sig = neuralNet.predict(X_test)
for i in range(0,len(y_pred_sig)):
    if(y_pred_sig[i]>0.5):
        y_pred_sig[i]=1
    else:
        y_pred_sig[i]=-1
    
cm = metrics.confusion_matrix(y_test,y_pred_sig)
print("Confusion Matrix:\n ",cm)
print( "{0}".format(metrics.classification_report(y_test,y_pred_sig)))
accuracy_test = metrics.accuracy_score(y_test,y_pred_sig)*100
print('Accuracy:%.2f' % accuracy_test,"%")

# Visualising the Training set results
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
fig1 = plot_decision_regions(X_train, y_train, clf=neuralNet, ax=axes[0], legend=0)
fig2 = plot_decision_regions(X_test, y_test, clf=neuralNet, ax=axes[1], legend=0)

axes[0].set_title('Sigmoid with Hidden Layers (Training set)', fontsize=10)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[1].set_title('Sigmoid with Hidden Layers (Test set)',fontsize=10)
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles, 
          ['class -1', 'class 1'])
fig2.legend(handles, 
          ['class -1', 'class 1'])

plt.tight_layout()
plt.show()


"""
Question 2(c):	
Implement the Pocket algorithm (or any improved Perceptron algorithm; there are many you can find 
online) and run it on your synthetic data set which you have modified in such a way that it is not
linearly separable anymore. Compare the Pocket with Perceptron, Adaline, and Sigmoid on the data 
and discuss your results.
"""
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=5000, shuffle=False, noise=None, random_state=None, factor=0.5)

#X, y = make_classification(n_samples=5000, n_features=2, n_redundant=0, n_informative=2,
#                             n_clusters_per_class=1,class_sep=0.7,flip_y=0.3, random_state=42,shuffle=False)

for i, j in enumerate(np.asarray(y)):
    if j==0:
        y[i] = -1
 
unique_elements, counts_elements = np.unique(y, return_counts=True)
print("Frequency of unique class of the array:")
print(np.asarray((unique_elements, counts_elements)))
print("\nVisualizing the synthetic dataset of Class 1 and Class -1: ")
plt.plot(X[:, 0][y == -1], X[:, 1][y == -1], 'g^', label='Class: -1')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'o', label="Class: 1")
plt.title("Dataset")
plt.xlabel("X1")
plt.xlabel("X2")
plt.legend()
plt.margins()
plt.show()  
   
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Perceptron to the Training set
from sklearn.linear_model import Perceptron 
classifier = Perceptron(random_state=42,alpha=0.01,eta0=0.1,max_iter=100)
classifier.fit(X_train, y_train)
weights = classifier.coef_
print('Weights before applying improvement algorithms:', weights)
#Applying GridSearch K-fold Cross Validation to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'alpha':[0.0001,0.001,.01,.1], 'fit_intercept':[True, False], 'eta0':[0.1,0.5,1]}]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, cv=10, scoring='accuracy')
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_parameters= grid_search.best_params_
print("After applying K-fold Cross validation and Grid Search Technique"
      +"\nthe best parameters are found as follows:")
print(best_parameters)
classifier = Perceptron (random_state=42,alpha=0.001,fit_intercept=False, eta0=1,max_iter=100)
classifier.fit(X_train, y_train)
weights = classifier.coef_
print('Weights after applying improvement algorithms:', weights)

print("Cross validation average accuracy and std of Perceptron on the Traing Set :") 
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv =10 )
acc = accuracies.mean()*100
print('Mean Accuracy: %.2f'% acc,'%' )
print("STD: ", accuracies.std())

print("Test set result: ")

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n ",cm)
print( "{0}".format(metrics.classification_report(y_test,y_pred)))
accuracy_test = metrics.accuracy_score(y_test,y_pred)*100
print('Accuracy:%.2f' % accuracy_test,"%")

# Visualising the Training set and Test set results
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
fig1 = plot_decision_regions(X_train, y_train, clf=classifier, ax=axes[0], legend=0)
fig2 = plot_decision_regions(X_test, y_test, clf=classifier, ax=axes[1], legend=0)

axes[0].set_title('Perceptron (Training set)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[1].set_title('Perceptron (Test set)')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles, 
          ['class -1', 'class 1'])
fig2.legend(handles, 
          ['class -1', 'class 1'])

plt.tight_layout()
plt.show()


#Fitting Adaline (SGD) to the training set
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier(random_state=42,max_iter=200)
classifier.fit(X_train, y_train)

print("Cross validation average accuracy and std of Adaline on the Traing Set:") 
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv =10 )
acc = accuracies.mean()*100
print('Mean Accuracy: %.2f'% acc,'%' )
print("STD: ", accuracies.std())

# Predicting the Test set results
y_pred = classifier.predict(X_test)

print("Test set result: ")
# Making the Confusion Matrix
cm = metrics.confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n ",cm)
print( "{0}".format(metrics.classification_report(y_test,y_pred)))
accuracy_test = metrics.accuracy_score(y_test,y_pred)*100
print('Accuracy:%.2f' % accuracy_test,"%")


#cm = confusion_matrix(y_test, y_pred)
#print( "The accuracy and confusion matrix of Perceptron on the Test Set")
#print("Confusion Matrix:\n", cm)
#precision = cm[1][1]/(cm[1][1]+cm[0][1]) #tp/tp+fp 
#recall = cm[1][1]/(cm[1][1]+cm[1][0]) #tp/tp+fn
#print("Precision: %.3f"% precision)
#print("Recall: %.3f" %recall)
#print('Accuracy: %.2f' % (classifier.score(X_test, y_test)*100),"%")

# Visualising the Training set and Test set results
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
fig1 = plot_decision_regions(X_train, y_train, clf=classifier, ax=axes[0], legend=0)
fig2 = plot_decision_regions(X_test, y_test, clf=classifier, ax=axes[1], legend=0)

axes[0].set_title('Adaline SGD (Training set)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[1].set_title('Adaline SGD (Test set)')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles, 
          ['class -1', 'class 1'])
fig2.legend(handles, 
          ['class -1', 'class 1'])

plt.tight_layout()
plt.show()

# Fitting Sigmoid model to the Training set
def create_neural_network():
    model = Sequential()
    model.add(Dense(units=1, input_dim=2, kernel_initializer='uniform', activation='sigmoid'))
    model.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
    return model

neuralNet = KerasClassifier(build_fn=create_neural_network, epochs=20, batch_size=500)
neuralNet.fit(X_train,y_train, epochs = 20, batch_size=500)
accuracies_neural= cross_val_score(estimator=neuralNet, X=X_train, y=y_train, cv =10 )
print("Cross validation average accuracy and std of Sigmoid Neuron on the Traing Set:") 
acc = accuracies_neural.mean()*100
print('Mean Accuracy: %.2f'% acc,'%' )
print("STD: ", accuracies_neural.std())

print("Test set result: ")
# Predicting the Test set results
from sklearn import metrics
y_pred_sig = neuralNet.predict(X_test)
for i in range(0,len(y_pred_sig)):
    if(y_pred_sig[i]>0.5):
        y_pred_sig[i]=1
    else:
        y_pred_sig[i]=-1
    
cm = metrics.confusion_matrix(y_test,y_pred_sig)
print("Confusion Matrix:\n ",cm)
print( "{0}".format(metrics.classification_report(y_test,y_pred_sig)))
accuracy_test = metrics.accuracy_score(y_test,y_pred_sig)*100
print('Accuracy:%.2f' % accuracy_test,"%")

#cm_sig = confusion_matrix(y_test, y_pred_sig)
#cm_sig = np.delete(cm_sig,0,axis=1)
#cm_sig = np.delete(cm_sig,1,axis=0)
#print( "The accuracy and confusion matrix of Sigmoid Neuron on the Test Set")
#print("Confusion Matrix:\n", cm_sig)
#precision = cm_sig[1][1]/(cm_sig[1][1]+cm_sig[0][1]) #tp/tp+fp 
#recall = cm_sig[1][1]/(cm_sig[1][1]+cm_sig[1][0]) #tp/tp+fn
#print("Precision: %.3f"% precision)
#print("Recall: %.3f" %recall)
#accuracy_test = float(cm_sig[0][0]+cm_sig[1][1])/(cm_sig[0][0]+cm_sig[0][1]+cm_sig[1][0]+cm_sig[1][1])*100
#print('Accuracy:%.2f' % accuracy_test,"%")

# Visualising the Training set results
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
fig1 = plot_decision_regions(X_train, y_train, clf=neuralNet, ax=axes[0], legend=0)
fig2 = plot_decision_regions(X_test, y_test, clf=neuralNet, ax=axes[1], legend=0)

axes[0].set_title('Sigmoid (Training set)')
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[1].set_title('Sigmoid (Test set)')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles, 
          ['class -1', 'class 1'])
fig2.legend(handles, 
          ['class -1', 'class 1'])

plt.tight_layout()
plt.show()

#after adding hidden layers to the same sigmoid neuron 
def create_neural_network():
    model = Sequential()
    model.add(Dense(units=20, input_dim=2, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=15, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile (optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
    return model

neuralNet = KerasClassifier(build_fn=create_neural_network, epochs=20, batch_size=500)
neuralNet.fit(X_train,y_train, epochs = 20, batch_size=500)
accuracies_neural= cross_val_score(estimator=neuralNet, X=X_train, y=y_train, cv =10 )
print("Cross validation average accuracy and std of Sigmoid Neuron on the Traing Set:") 
acc = accuracies_neural.mean()*100
print('Mean Accuracy: %.2f'% acc,'%' )
print("STD: ", accuracies_neural.std())

print("Test set result: ")
# Predicting the Test set results
y_pred_sig = neuralNet.predict(X_test)
for i in range(0,len(y_pred_sig)):
    if(y_pred_sig[i]>0.5):
        y_pred_sig[i]=1
    else:
        y_pred_sig[i]=-1
 
cm = metrics.confusion_matrix(y_test,y_pred_sig)
print("Confusion Matrix:\n ",cm)
print( "{0}".format(metrics.classification_report(y_test,y_pred_sig)))
accuracy_test = metrics.accuracy_score(y_test,y_pred_sig)*100
print('Accuracy:%.2f' % accuracy_test,"%")

# Visualising the Training set results
fig, axes = plt.subplots (nrows=1, ncols=2, figsize=(8, 4))
fig1 = plot_decision_regions(X_train, y_train, clf=neuralNet, ax=axes[0], legend=0)
fig2 = plot_decision_regions(X_test, y_test, clf=neuralNet, ax=axes[1], legend=0)

axes[0].set_title('Sigmoid with Hidden Layers (Training set)', fontsize=10)
axes[0].set_xlabel('x1')
axes[0].set_ylabel('x2')
axes[1].set_title('Sigmoid with Hidden Layers (Test set)',fontsize=10)
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')

handles, labels = fig1.get_legend_handles_labels()
fig1.legend(handles, 
          ['class -1', 'class 1'])
fig2.legend(handles, 
          ['class -1', 'class 1'])

plt.tight_layout()
plt.show()

"""
Question #3: We now that the XOR or XNOR functions are not linearly separable functions. 
How would you modify the Perceptron learning algorithm in order to learn the XOR or XNOR function?
"""

import numpy as np

def activation(value):
	if value >= 0:
		return 1
	else:
		return 0

def perceptron(x, w, b):
    value = np.dot(w,x) + b
    y = activation(value)
    return y

def OR_perceptron(x):
    w = np.array([1, 1])
    b = -0.5
    return perceptron(x, w, b)

def AND_perceptron(x):
    w = np.array([1, 1])
    b = -1.5
    return perceptron(x, w, b)

def NOT_perceptron(x):
    if x == 1:
        return 0
    else:
        return 1
    
def XOR_net(x):
    gate1 = OR_perceptron(x)
    gate2 = AND_perceptron(x)
    gate3 = NOT_perceptron(gate2)
    x_update = np.array([gate1,gate3])
    output = AND_perceptron(x_update)
    return output

def XNOR_net(x):
    gate1 = OR_perceptron(x)
    gate2 = AND_perceptron(x)
    gate3 = NOT_perceptron(gate2)
    x_update = np.array([gate1,gate3])
    output = AND_perceptron(x_update)
    output = NOT_perceptron(output)
    return output

input1 = np.array([1, 1])
input2 = np.array([1, 0])
input3 = np.array([0, 1])
input4 = np.array([0, 0])

print("XOR({}, {}) = {}".format(1, 1, XOR_net(input1)))
print("XOR({}, {}) = {}".format(1, 0, XOR_net(input2)))
print("XOR({}, {}) = {}".format(0, 1, XOR_net(input3)))
print("XOR({}, {}) = {}".format(0, 0, XOR_net(input4)))
print("\n")
print("XNOR({}, {}) = {}".format(1, 1, XNOR_net(input1)))
print("XNOR({}, {}) = {}".format(1, 0, XNOR_net(input2)))
print("XNOR({}, {}) = {}".format(0, 1, XNOR_net(input3)))
print("XNOR({}, {}) = {}".format(0, 0, XNOR_net(input4)))


