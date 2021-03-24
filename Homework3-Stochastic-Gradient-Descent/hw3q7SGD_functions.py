#class   : Foundations of ML and Data Science
#homework: 3
#question: 7, function file 
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

import numpy as np


def generate_random_matrix(mu,sigma,d):
    # n - Number of data points
    # d - Number of dimensions
    # mu- Mean of the Gaussian variables vector in each row vector N X 1
    #sigma - Standard deviation of Gaussian variables in each row vector N X 1
    #Making n X d random matrix
    return np.array([np.random.normal(m, s, d) for m,s in zip(mu, sigma)])

def theoretical_conv(A,b,x_opt,x_initial_guess,max_iters,L):
    # A - n X d normal matrix
    # b - n X 1 RHS matrix
    # x_opt - optimal value of d X 1
    # x_initial_guess - initial guess of size d X 1
    ATA_eig, _ = np.linalg.eig(np.transpose(A) @ A)
    
    mu = np.min(ATA_eig) #lowest EVal

    sigma_sq = 0 #initialize    
    n = np.shape(A)[0]
    
    for i in range(n):
        sigma_sq = sigma_sq + np.linalg.norm(A[i,:])**2.0 * ( A[i,:] @ x_opt - b[i] )**2.0
        
    sigma_sq = sigma_sq * n
    
    expct_err_norm_sq = np.zeros((max_iters))
    expct_err_norm_sq[0] = np.linalg.norm(x_initial_guess-x_opt)**2.0   
    
    for t in range(1,max_iters):
        expct_err_norm_sq[t] = (1-mu/(2*L))**t * expct_err_norm_sq[0] + sigma_sq/(mu*L)
        
    return expct_err_norm_sq

def stochastic_index_compute(n,max_iters):
    stochastic_index = np.random.permutation(n)
    if(max_iters%n != 0 and int(max_iters/n) != 0):
        stochastic_index = np.concatenate(np.tile(stochastic_index, int(max_iters/n)), stochastic_index[:(max_iters%n)])
    elif(max_iters%n == 0):
        stochastic_index = np.tile(stochastic_index, int(max_iters/n))
    elif(int(max_iters/n) == 0):
        stochastic_index = stochastic_index[:(max_iters%n)]
    else:
        print("Error!")   
    return stochastic_index