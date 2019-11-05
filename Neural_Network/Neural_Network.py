import random
import numpy as np
import math
import random
import sys
import pandas as pd
from operator import add

train = sys.argv[1]
test = sys.argv[2]
m = int(sys.argv[3])                    
eta = float  (sys.argv[4])
epochs = int(sys.argv[5])

train = pd.read_csv(train, header=None)
test = pd.read_csv(test, header=None)
train = np.array(train)
test = np.array(test)                       

n_test, d_test = test.shape
n, d = train.shape

# Removing x and y from the test set
Xtest = test[:, 0: d_test-1]
Ytest = test[:, d_test-1]

# Removing x and y from the traning set               
Xtrain = train[:, 0: d-1]
Ytrain = train[:, d-1]

unique = list(np.unique(Ytrain))
unique_len = len(unique)
print(unique)

# one hot encoding the training response variable
hot_encoded_y  = []
for  i in range(len(Ytrain)):
    temp = [0]* unique_len
    temp[unique.index(Ytrain[i])] = 1
    hot_encoded_y.append(temp)
    if(temp == 1):
        print(temp)

# one hot encoding the testing response variable
hot_encoded_y = np.array(hot_encoded_y)
hot_encoded_y_test = []
for i in range(len(Ytest)):
    temp = [0]*unique_len
    temp[unique.index(Ytest[i])] = 1
    hot_encoded_y_test.append(temp)
hot_encoded_y_test = np.array(hot_encoded_y_test)
r = list(range(n))
random.shuffle(r)
p= unique_len

#defining the MLP_train function
def MLP_train(D, m, eta, epochs):
    b_h = pd.DataFrame([0.5]*m)
    b_o = pd.DataFrame([0.5] * p)
    w_h =np.array([[0.01 for col in range(m)]
              for row in range(d-1)], dtype=float)
    w_o = np.array([[0.01 for col in range(p)]
                    for row in range(m)], dtype=float)
    t =0

    # calculating net at the Output Layer
    while(t<epochs):        
        for i in r:
            # feed forward
            net_z = np.array (b_h + np.dot(w_h.T, D[i]).reshape((-1,1)))
            for j in range(len(net_z)):
                net_z[j] = float( max(0,net_z[j]))

            # Calculating the next layer z to o
            #SOFTMAX
            net_o = np.array(b_o + np.dot(w_o.T, net_z))
            total =0
            for j in range(len(net_o)):
                total += math.exp(net_o[j][0])
            o = np.divide ( net_o,  total) # chanhge this 

            #Backpropagation step
            derivative_relu =[0]  * len(net_z)
            for j in range(len(net_z)):
                if(net_z[j] > 0):
                    derivative_relu[j] = 1

            delta_o = np.array([y-x for x, y in zip(hot_encoded_y[i], o)])
            delta_h =np.array([ sum(x) for x in zip(
                list(derivative_relu), list(np.dot(w_o, delta_o)))])

            #Performing Gradient descent on bias vectors
            delta_bo = delta_o
            b_o = b_o - eta*delta_bo
            delta_bh = delta_h
            b_h = b_h - eta*delta_bh

            #Performing Gradient descent on weight matrices
            delta_wo = np.dot(net_z,delta_o.T)
            w_o = w_o - eta*delta_wo
            delta_wh = np.dot(delta_h, D[i].reshape((1,d-1)))
            w_h = w_h - eta*delta_wh.T

        # softmax
        print("iteration", t)
        t+=1
    return w_h,w_o, b_h,b_o
w_h, w_o, b_h, b_o = MLP_train(Xtrain, m, eta, epochs)

print("b hidden")
print(pd.DataFrame(b_h))

print("w hidden")
print(pd.DataFrame(w_h))

print("b output")
print(pd.DataFrame(b_o))

print("w output")
print(pd.DataFrame(w_o))

#Calculating accuracy
count = 0
for i in range(n_test):
    net_z = np.array(b_h + np.dot(w_h.T, Xtest[i]).reshape((-1, 1)))
    for j in range(len(net_z)):
         net_z[j] = float(max(0, net_z[j]))

    #SOFTMAX
    net_o = np.array(b_o + np.round( np.dot(w_o.T, net_z), 9))
    total = 0
    for j in range(len(net_o)):
        total += math.exp(net_o[j][0])
    o = np.divide(net_o,  total)  
    
    check = []
    for j in range(len(o)):
        check.append(float(o[j]))
    check = np.array (check)   
    max_val =  max(check)
    for j in range(len(check)):
        if ( check[j] == max_val):
            check[j]= 1
        else :
            check[j] =0 
    if(np.array_equal (check, hot_encoded_y_test[i])):
        count+=1

print("accuracy = ", float(count/n_test) *100 ,"%")
