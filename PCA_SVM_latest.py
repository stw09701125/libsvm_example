#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:25:23 2017

@author: stw
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Train_data = pd.read_csv('X_train.csv', header = None)
Train_label = pd.read_csv('T_train.csv', header = None)
Test_data = pd.read_csv('X_test.csv', header = None)
Test_label = pd.read_csv('T_test.csv', header = None)

X_train = Train_data.iloc[:, :].values
Y_train = Train_label.iloc[:].values
X_test = Test_data.iloc[:, :].values
Y_test = Test_label.iloc[:].values


def plot(dataset, labels, titlename=''):
    colors = ["skyblue", "lightgray", "greenyellow", "salmon", "khaki"]
    for index in range(1, 6):
        plt.scatter(dataset[labels[:, 0] == index, 0], dataset[labels[:, 0] == index, 1], s = 30, c = colors[index - 1], label = 'Cluster ' + str(index))
    plt.title('PCA of MNIST ' + titlename)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()
        
def change_format(X, Y, dimension, name):
    svm_format = np.concatenate((Y, X), axis=1)
    svm_format = svm_format.astype(str)
    for item in svm_format:
        for idx in range(1, dimension + 1):
            item[idx] = str(idx) + ":" + item[idx]
    np.savetxt(name, svm_format, fmt="%s")

# Applying PCA
from sklearn.decomposition import PCA

D = 2
pca = PCA(D)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
change_format(X_train_pca, Y_train, D, "svm_format_train")
change_format(X_test_pca, Y_test, D, "svm_format_test")

        
from svmutil import *

y, x = svm_read_problem("svm_format_train")

model = svm_train(y, x, "-c 2 -g 0.125")
y_t, x_t = svm_read_problem("svm_format_test")

p_label, p_acc, p_val = svm_predict(y_t, x_t, model)

def find_support_vector(model, training_set):
    nSV = [] # number of support vectors of each label
    SVs = [] # index of each support vector in trainging data
    support_v = [] # the support vectors
    begin = 0
    end = 0
    
    # store number of support vectors
    for i in range(5):
        nSV.append(model.nSV[i])
    # find out the index of each support vector in training data 
    for num in nSV:
        temp = []
        end += num
        for idx in range(begin, end):
            temp.append(model.sv_indices[idx] - 1)
        begin = end
        SVs.append(temp)
    # select every support vector in dataset 
    for item in SVs:
        support_v.append(training_set[item])
    
    return support_v
    
def plot_with_SV(dataset, labels, SV, titlename=''):
    colors = ["skyblue", "lightgray", "greenyellow", "salmon", "khaki"]
    colors_v = ["rebeccapurple", "red", "k", "b", "limegreen"]
    for index in range(1, 6):
        plt.scatter(dataset[labels[:, 0] == index, 0], dataset[labels[:, 0] == index, 1], s = 30, c = colors[index - 1], label = 'Cluster ' + str(index))
        plt.scatter(SV[index - 1][0:10, 0], SV[index - 1][0:10, 1], s = 100, c = colors_v[index - 1], marker=(5, 1))
    plt.title('PCA of MNIST ' + titlename)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()

SV = find_support_vector(model, X_train_pca)

plot_with_SV(X_test_pca, np.array(p_label).reshape(-1, 1), SV, 'predictive data')

# plot decision boundary
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train_pca, Y_train.reshape(Y_train.shape[0]))

y_pred = classifier.predict(X_test_pca)

from matplotlib.colors import ListedColormap
X_set, Y_set = X_test_pca, Y_test.reshape(Y_test.shape[0])
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.8, cmap = plt.cm.rainbow)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('rebeccapurple', 'royalblue', 'limegreen', 'sandybrown', 'red'))(i), label = j)
plt.title('SVM (Testing set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
