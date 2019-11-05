import sys
import math as math
import pandas as pd
import numpy.linalg as LA
import numpy as np
import math as math

D = np.loadtxt(sys.argv[1],delimiter=',')

#Removing the response variable from the dataset
n,d = np.shape(D)
Y = D[:,d-1]
D1 = np.delete(D, d-1, axis=1)
X0 = np.ones((n,1))

#Augmenting the Data Matrix
aug_D = np.hstack((X0,D1))
n1,d1 = np.shape(aug_D)

#sigmoid function
def theta(z):           
    return 1/(1+math.exp(-z))    
w0 = np.zeros((1,d1))
w_t = w0
w = w_t
test  = sys.argv[2]
eta = sys.argv[4]
epsilon = sys.argv[3]

#logistic regression algorithm:SGA
while True:
    w = w_t
    for i in range(0,n1):
        delta_wi_xi = np.dot(Y[i] - theta(np.dot(w, aug_D[i,:].T)),aug_D[i,:])
        w = w + float(eta)*delta_wi_xi  
    w_tplusone = w 
    if(LA.norm(w_tplusone - w_t) <= float(epsilon)):
        break 
    else:
        w_t = w_tplusone    

print("epsilon=",epsilon)
print("eta=", eta)             
print("The weight vector is:\n")
print(pd.DataFrame(w_tplusone))

#testing our model on the test dataset
test = np.loadtxt(sys.argv[2],delimiter=',')
n,d = np.shape(test)
Y = test[:,d-1]
D1 = np.delete(test, d-1, axis=1)
X0 = np.ones((n,1))

#Augmenting the Data Matrix
aug_D = np.hstack((X0,D1))
n2,d2 = np.shape(aug_D)

#testing our model on the test dataset and calculating the accuracy of the model
correct = 0
incorrect = 0
for i in range(0,n2):
    sigmoid1 = theta(np.dot(w, aug_D[i,:].T))
    if(float(sigmoid1) >= 0.5):
        pred = 1
    else :
        pred  =  0
    if( pred == Y[i]):
        correct += 1  
    else:
        incorrect += 1
print("accuracy = ", float(correct/n2)*100, "%")