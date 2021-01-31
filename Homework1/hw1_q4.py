#class   : Foundations of ML and Data Science
#homework: 1
#question: 4(a)
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#Power method
import numpy as np

tol    = 1e-14    #tolerance for power method
max_iter= 5000    #maximum number of iterations
n      = 10       #size
A      = np.zeros((n,n)) #placeholder for A matrix

for i in range(0,n):
    for j in range(0,n):
        if i + j < n : 
            A[i,j] = i + j + 1

print(A) 

v      = np.random.rand(n,1) #randomly choosing a vector v
v      = v/np.linalg.norm(v) #normalize v
v_old  = np.ones_like(v)    #placeholder for old vector v

i      = 0

#power method
while (np.linalg.norm(v-v_old)/np.linalg.norm(v))>tol and i < max_iter: #relative error
    v_old = np.copy(v)
    v     = A @ v_old             #finding new vector
    v     = v/np.linalg.norm(v)   #normalize
    lmbda = (np.matrix.transpose(v) @ (A @ v))[0,0]     
    i = i + 1
print(i,lmbda,v)   

U,D,V =  np.linalg.svd(A)
