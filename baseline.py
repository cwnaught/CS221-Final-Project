# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:13:57 2019

@author: Chris Naughton
"""
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

trainData = np.loadtxt('trainData.csv',delimiter=',')
testData = np.loadtxt('testData.csv',delimiter=',')
trainTruth = np.loadtxt('trainTruth.csv',delimiter=',')
testTruth = np.loadtxt('testTruth.csv',delimiter=',')

clf = svm.SVC(gamma = 'scale', decision_function_shape='ovo')
clf.fit(trainData,trainTruth)

guess = clf.predict(testData)

correct = guess==testTruth
perc = np.sum(correct) / len(correct) * 100

print('Baseline Accuracy = ' + str(perc) + '%')

print(confusion_matrix(testTruth,guess))