#class   : Foundations of Data Science
#homework: 1
#question: 4(c)
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#Block Power method
import numpy as np
from scipy.linalg import orth 

tol    = 1e-15    #tolerance for power method
max_iter= 5000    #maximum number of iterations
n      = 10       #size
A      = np.zeros((n,n)) #placeholder for A matrix
s      = 8        #number of singular vectors

for i in range(0,n):
    for j in range(0,n):
        if i + j < n : 
            A[i,j] = i + j + 1

print('A: \n',A,'\n') 

v      = np.random.rand(n,s) #randomly choosing a vector v
err    = 1.0
i      = 0

#block power method
while err > tol and i < max_iter:  #error comparison
    v_old = orth(v)
    v     = A @ v_old      #finding new matrix B =Av
    err   = np.linalg.norm(v - v_old)
    i = i + 1

v     = v/np.linalg.norm(v, ord=2, axis=0, keepdims=True) #normalize each column

lmbda = np.matrix.transpose(v) @ (A @ v)                  #finding Rayleigh-quotients

for i in range(0,s):
    if lmbda[i,i] < 0.0: #for negative singular values, change the sign
        lmbda[i,i] = np.abs(lmbda[i,i])
        v[:,i]     = -v[:,i] 

print('Iteration: ',i,'\n','V: \n',v)
print('Singular values', np.diag(lmbda))
#Verify
U,D,VT =  np.linalg.svd(A)
V_direct = np.transpose(VT[0:s,:])

print('Frobenius norm of error in the V matrices', np.linalg.norm(np.absolute(v) - np.absolute(V_direct)))