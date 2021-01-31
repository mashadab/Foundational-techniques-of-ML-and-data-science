#class   : Foundations of Data Science
#homework: 1
#question: 4(c)
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#Block Power method
import numpy as np

tol    = 1e-14    #tolerance for power method
max_iter=5000     #maximum number of iterations
n      = 10       #size
A      = np.zeros((n,n)) #placeholder for A matrix
s      = 4        #number of singular vectors

for i in range(0,n):
    for j in range(0,n):
        if i + j < n : 
            A[i,j] = i + j + 1

print('A: \n',A,'\n') 

v      = np.random.rand(n,s) #randomly choosing a vector v
err    = 1.0
i      = 0

#block power method
while err > tol and i < max_iter: #error comparison
    B     = A @ v                 #finding new matrix B =Av
    Q,R   = np.linalg.qr(B)       #finding QR factorization of B
    v     = Q[:,0:s]              #extraction of orthonormal vectors close to singular vectors
    lmbda = R[0:s,:]              #extraction of singular values (can be negative)
    err   = np.linalg.norm(A@v - v @ lmbda)
    i = i + 1
   
print('Iteration: ',i,'\n','Lambda: \n',lmbda,'\n','V: \n',v)

U,D,V =  np.linalg.svd(A)
