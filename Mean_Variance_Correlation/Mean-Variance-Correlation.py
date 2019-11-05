#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 21:53:16 2019

@author: abhishekgupta
"""

import numpy as np
import chart_studio.plotly as cs
from numpy import linalg as la
from numpy.linalg import eig
import statistics as st
import matplotlib.pyplot as mpl
import pandas as pd
import seaborn as sns

dataset = np.loadtxt( '/Users/abhishekgupta/Desktop/airfoil_self_noise.dat' )
matrix = np.loadtxt( '/Users/abhishekgupta/Desktop/airfoil_self_noise.dat' )
test = np.loadtxt( '/Users/abhishekgupta/Desktop/airfoil_self_noise.dat' )
D = np.loadtxt( '/Users/abhishekgupta/Desktop/airfoil_self_noise.dat' )
sum = []

epsilon = float(input("Enter the threshold epsilon:"))

#calculating mean
mean=[]
sumofmatrices = []
centredmatrix = []
dimension=np.shape(dataset)
print(dimension)
columnsum = np.sum(dataset, axis=0)
mean = columnsum/dimension[0]
print("The mean of each attribute is:")
print(mean)
print("----------------------------------------------------------------")


#claculating variance
for i in range(0,(dimension[1])):
    dataset[:,i] = (dataset[:,i] - mean[i])**2 
    columnsum1 = np.sum(dataset, axis=0)
    variance = columnsum1/dimension[0]    
print("The variance of each attribute is:")
print(variance)    
print("Tht total variance is:", np.sum(variance))
print("----------------------------------------------------------------")


#claculating covariance through inner product
centeredmatrix = matrix - (np.mean(matrix, axis=0))
transpose = np.transpose(centeredmatrix)
covariance = (np.dot(transpose,centeredmatrix))/dimension[0]
print("The covariance matrix through inner product is:")
print(covariance)
print("----------------------------------------------------------------")


#calculating covariance through outer product
for i in range(dimension[0]):
    temp = np.array(centeredmatrix[i])
    reshapedmatrix = temp.reshape((-1,1))
    if(i==0):
        outercov = np.dot(reshapedmatrix, reshapedmatrix.T) /dimension[0]
    else:
        outercov+=np.dot(reshapedmatrix, reshapedmatrix.T) /dimension[0]
        
print("The covraince matrix through outer product is:")
print(outercov)
print("----------------------------------------------------------------")

#calculating correlation 
corr= np.empty(covariance.shape)
for i in range(dimension[1]):
    for j in range(dimension[1]):
        
        z1= transpose[i].reshape((-1,1))
        
        z2= transpose[j].reshape((-1,1))
        
        corr[i][j] = corr[j][i] = np.dot( z1.T / la.norm(z1),z2 / la.norm(z2))
print("The correlation matrix is:")
print(corr)
print("----------------------------------------------------------------")

#mpl.scatter(centeredmatrix[:,1],centeredmatrix[:,2])
sns.pairplot(pd.DataFrame(corr))

print("----------------------------------------------------------------")

#part2
Xn_i = [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]]
while (1):
    Xn = np.dot(covariance,Xn_i)                            #X1 = sigma*X0 
    a = Xn[:,0]
    b = Xn[:,1]
    b = b - np.dot(((np.dot(b.T,a))/(np.dot(a.T,a))),a)   #orthogonalise b
    Xn[:,0] = Xn[:,0]/max(Xn[:,0])
    Xn[:,1] = Xn[:,1]/max(Xn[:,1])
    dist = np.linalg.norm(Xn - Xn_i)
    if(epsilon<=dist):
        Xn_i = Xn
    else:
        break;

print("The eigen values are:")
print("The eigen vectors u1 and u2 are:")
print(Xn)
print("----------------------------------------------------------------")

cm = D - (np.mean(D, axis=0))
p = np.dot(cm,Xn)   #orthogonalise b
print("The new projected points in 2D are:")
print(p)
mpl.scatter(p[:,0],p[:,1])

print("----------------------------------------------------------------")

















