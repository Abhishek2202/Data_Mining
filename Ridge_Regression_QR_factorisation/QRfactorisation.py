import numpy as np
import sys
import math as math
import matplotlib.pyplot as plt
print ("Output:")

D = np.loadtxt(sys.argv[1],delimiter=',')
print("\nFor file: "+str(sys.argv[1]))

#Removing out the response variable
Y = D[0:,-1]
n = np.shape(D)[0]
Y = np.reshape(Y, (n,1))
D1 = np.delete(D, -1, axis=1)
X0 = np.ones((n,1))

# Augmenting the matrix
D0 = np.hstack((X0,D1))
d = np.shape(D0)[1]

A=np.zeros((d,d))
for i in range(0,d):
	for j in range(0,d):
		if i==j:
			A[i,j]=1

a=float(sys.argv[3])
print("Entered Value of ridge constant is: "+str(sys.argv[3]))
A = (math.sqrt(a))*A
D0 = np.vstack((D0,A))

Z=np.zeros((d,1))
Y=np.vstack((Y,Z))

#Calculating Q
Q = np.zeros((n,d))
t = np.zeros((d,d))
Q = np.vstack((Q,t))
Q[0:,0] = 1
for i in range(1, d):
	Q[0:,i]= D0[0:,i]
	for j in range(i):
		Q[0:,i] = Q[0:,i] - ((np.dot(D0[0:,i],Q[0:,j])/pow(np.linalg.norm(Q[0:,j]) , 2)) * Q[0:,j])

#Calculating delta inverse
deltainv = np.zeros((d,d))
for i in range(d):
	for j in range(d):
		if i==j:
			deltainv[i,j] = 1/(pow(np.linalg.norm(Q[0:,i]),2))
			break

#Calculating R
R=np.zeros((d,d))
for i in range(d):
	for j in range(d):
		if i==j:
			R[i,j]=1
		if i<j:
			R[i,j] = (np.dot(D0[0:,j],Q[0:,i])) / (pow(np.linalg.norm(Q[0:,i]),2))

#Calculating weight vector
M = np.dot((np.dot(deltainv,Q.T)),Y)

def back_sub(A:np.ndarray, b:np.ndarray) -> np.ndarray:
    
    n1 = b.size
    w = np.zeros_like(b)

    if A[n1-1, n1-1] == 0:
        raise ValueError


    w[n1-1] = b[n1-1]/A[n1-1, n1-1]
    C = np.zeros((n1,n1))
    for i in range(n1-2, -1, -1):
        bb = 0
        for j in range (i+1, n1):
            bb += A[i, j]*w[j]

        C[i, i] = b[i] - bb
        w[i] = C[i, i]/A[i, i]
    return w

w=back_sub(R,M)
print("\nThe weight vector w using ridge regression is:")
print(w)

#Calculating L norm of the weight vector
print("\nL2 norm of weight vector:")
print(np.linalg.norm(w))

#Calculating the SSE and R square for the training dataset
Ycap = np.dot(Q,M)
E=Y-Ycap
SSE = np.sum(np.square(E))
meanY = (np.sum(Y))/n
TSS = np.sum(np.square(Y-meanY))
Rsquare = (TSS-SSE)/TSS 
print("\nThe Value of SSE for the training dataset is:")
print(SSE)
print("\nThe Value of R Square for the training dataset is:")
print(Rsquare)

#Applying our model on testing dataset
D = np.loadtxt(sys.argv[2],delimiter=',')
print("\nTesting our model on the testing dataset")
Y = D[0:,-1]
n = np.shape(D)[0]
D1 = np.delete(D, -1, axis=1)
X0 = np.ones((n,1))
D0 = np.hstack((X0,D1))

#Calculating the SSE and R square values for the testing dataset
Ycap=(w[0]*D0[0:,0])
for i in range(1,d):
	Ycap=Ycap+(w[i]*D0[0:,i])

E=Y-Ycap
SSE = np.sum(np.square(E))
meanY = (np.sum(Y))/n
TSS = np.sum(np.square(Y-meanY))
Rsquare = (TSS-SSE)/TSS

print("\nThe value of SSE for testing data is::")
print(SSE)
print("\nThe value of R Square for testing data is:\n")
print(Rsquare)
