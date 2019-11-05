import numpy as np
import sys
import pandas as pd
import math
import numpy.linalg as LA


#defining the linear, quadratic and gaussian kernels
def linear (x,y):                                           
    return np.dot(x,y.T)

def quadratic(x,y):
     return (1 + np.dot(x,y.T)) ** 2

def guassian(x,y,spread):
    n1  = np.shape(x)[0]
    n2 = np.shape(y)[0]
    kernel = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            kernel[i][j]= math.exp(-(pow(LA.norm(x[i,:] - y[j,:]),2))/(2 * (float(spread) ** 2)))    
    return kernel

train = sys.argv[1]
test = sys.argv[2]
kernel = sys.argv[3]
try:
    spread = sys.argv[4]
    print("For ", kernel)
except IndexError:
    print("For " , kernel, "kernel")

alpha=0.01
train = np.loadtxt(sys.argv[1],delimiter=',')
test = np.loadtxt(sys.argv[2],delimiter=',')

n,d = np.shape(train)
n1,d1 = np.shape(test)
X0 = np.ones((n,1))

#separating response variables for train and test                       
Ytrain = train[:,d-1]                                                   
Ytest = test[:,d-1]                                                    
                                                                        
#removing the response variable from the dataset                        
D_train = np.delete(train, d-1, axis=1)
D_test = np.delete(test, d-1, axis=1)

#calculating Kernel Matrix K
if (kernel == "linear"):                                                
    K = np.add(linear(D_train, D_train),1)                                   
    Ktest = np.add(linear(D_test, D_train),1)

elif (kernel =="quadratic"):
    K = np.add(quadratic(D_train, D_train),1)
    Ktest = np.add(quadratic(D_test, D_train),1)

elif (kernel == "gaussian"):
    print("spread=", spread)
    K = np.add(guassian(D_train, D_train, spread),1)
    Ktest = np.add(guassian(D_test, D_train, spread),1)

else:
    print("invalid kernel type")
    sys.exit()

K = np.array(K)
alpha_I = np.dot(alpha,np.identity(n))                          
c = np.dot(LA.inv(K + alpha_I), Ytrain)
Ycap = np.dot(K, c)
                                                               
# testing the algorithm on training set
correct =0
for i in range(n):
    if(Ycap[i] >= 0.5):
        prediction = 1
    else:
        prediction = 0    
    if(prediction == Ytrain[i]):
        correct+= 1
print("training accuracy is:",float(correct/n)*100,"%")

# testing the algorithm on testing set
Ytest_pred = np.dot(Ktest, c)
correct = 0                                                     
incorrect =0
for i in range(n1):
    if(Ytest_pred[i] >= 0.5):
        prediction = 1
    else:
        prediction = 0                                
    if(prediction == Ytest[i]):
        correct += 1                                    
    else:
        incorrect+=1    
print("testing accuracy is:", float(correct/n1)*100, "%")