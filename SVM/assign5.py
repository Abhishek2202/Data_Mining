import numpy as np
import sys
import math as math
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as LA


#defining the linear, quadratic and gaussian kernels
def linear (x,y):                                           
    return np.dot(x,y.T)

def linear_phi_x(A):
    return A    
 
def quadratic(x,y):
     return (np.dot(x,y.T)) ** 2

def quadratic_phi_x(A):
    phi_x = pd.DataFrame()
    for i in range(A.shape[1]):
        temp = (A[:, i]) ** 2
        phi_x.insert(loc=i, column="x"+str(i+1), value=temp)
    k = i
    sq_root_2 = math.sqrt(2)
    for i in range(A.shape[1]):
        for j in range(i+1, A.shape[1], 1):
            temp = sq_root_2 * (A[:, i]) * A[:, j]
            phi_x.insert(loc=k, column="x"+str(i+1) + str(j+1), value=temp)
            k += 1
    # adding column of ones for bias
    temp = np.ones((n,1))
    phi_x.insert(loc=k, column="b", value=temp)
    return phi_x

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
C = float (sys.argv[3])
eps = float( sys.argv[4])
kernel = sys.argv[5]
try:
    spread = sys.argv[6]
    print("For ", kernel, "kernel")
except IndexError:
    print("For ", kernel, "kernel")

train = np.loadtxt(sys.argv[1],delimiter=',')
test = np.loadtxt(sys.argv[2],delimiter=',')

n,d = np.shape(train)
n1,d1 = np.shape(test)
X0 = np.ones((n,1)) 

#Separating response variables for train and test                       
Ytrain = train[:,d-1]                                                   
Ytest = test[:,d-1]  

#removing the response variable from the dataset                        
D_train = np.delete(train, d-1, axis=1)
D_test = np.delete(test, d-1, axis=1)

#Calculating Kernel Matrix K
if (kernel == "linear"):                                                
    K = np.add(linear(D_train, D_train),1)                                   
    Ktest = np.add(linear(D_test, D_train),1)
    ones_column = np.ones((n, 1))
    D_train = np.hstack((D_train, ones_column))
    phi_x = D_train

elif (kernel =="quadratic"):
    K = np.add(quadratic(D_train, D_train),1)
    Ktest = np.add(quadratic(D_test, D_train),1)
    phi_x = quadratic_phi_x(D_train)

elif (kernel == "gaussian"):
    print("spread=", spread)
    K = np.add(guassian(D_train, D_train, spread),1)
    Ktest = np.add(guassian(D_test, D_train, spread),1)

else:
    print("invalid kernel type")
    sys.exit()

eta_k = np.zeros((n,1))
for k in range(n):
    eta_k[k] = float(1/(K[k][k]))

t = 0
alpha0 = np.zeros((n,1))
alpha_t = alpha0
while True:
    alpha = alpha_t  
    for k in range(n):     
        sum = 0   
        for i in range(n):
            sum = sum + (alpha[i]*Ytrain[i]*K[i][k])
        alpha[k] = alpha[k] + eta_k[k]*(1 - float(Ytrain[k])*float(sum[0])) 
        if(alpha[k] < 0):
            alpha[k] = 0
        elif(alpha[k] > C):
            alpha[k] = C
    alpha_tplusone = alpha
    t = t+1
    if(LA.norm(alpha_tplusone - alpha_t) <= eps):
        break
    else:
        alpha_t = alpha_tplusone
print("The alpha values are:")
print(pd.DataFrame(alpha))

#printing the support vectors
print("The support vectors are:")
count = 0
for i in range(n):
    if(alpha_tplusone[i] >0):
        count = count +1
        print(i, "   " , float(alpha_tplusone[i]))
print("\nTotal number of Support Vectors:", count)

#testing the accuracy on the test set
correct = 0
for z in range(n1):
    sum = 0
    for i in range(n):
        sum = sum + (alpha_tplusone[i]*Ytrain[i]*Ktest[z][i])
    if(sum>0):
        Ypred = 1
    elif(sum<0):
        Ypred = -1
    else:
        Ypred = Ytest[z]
    if(Ypred == Ytest[z]):
        correct = correct + 1
print("\nAccuracy on training dataset is:", float((correct/n1)*100), "%\n")

#calculating the weight vector for the hyperplane
if (kernel =="linear"  or kernel == "quadratic"):
    phi_x = np.array(phi_x)
    w = np.zeros(phi_x.shape[1])
    for i in range(n):
        sum = np.multiply(alpha_tplusone[i]* Ytrain[i], phi_x[i])
        w = np.add (w, sum)
    print("The weight vector is:")
    print(w)    




