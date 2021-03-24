#class   : Foundations of ML and Data Science
#homework: 3
#question: 7(a),(b)
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#Stochastic Gradient Descent
import numpy as np
import matplotlib.pyplot as plt

def random_matrix(mu,sigma,d):
    # n - Number of data points
    # d - Number of dimensions
    # mu- Mean of the Gaussian variables vector in each row vector N X 1
    #sigma - Standard deviation of Gaussian variables in each row vector N X 1
    
    #Making n X d random matrix
    return np.array([np.random.normal(m, s, d) for m,s in zip(mu, sigma)])

def theoretical_convergence(A,b,x_optimal,x_initial_guess,max_iters,L):
    
    # A - n X d normal matrix
    # b - n X 1 RHS matrix
    # x_optimal - optimal value of d X 1
    # x_initial_guess - initial guess of size d X 1
    
    ATA_eig, _ = np.linalg.eig(np.transpose(A) @ A)
    
    mu = np.min(ATA_eig) #lowest EVal

    sigma_sq = 0 #initialize    
    n = np.shape(A)[0]
    
    for i in range(n):
        sigma_sq = sigma_sq + np.linalg.norm(A[i,:])**2.0 * ( A[i,:] @ x_optimal - b[i] )**2.0
        
    sigma_sq = sigma_sq * n
    
    expct_err_norm_sq = np.zeros((max_iters))
    expct_err_norm_sq[0] = np.linalg.norm(x_initial_guess-x_optimal)**2.0   
    
    for t in range(1,max_iters):
        expct_err_norm_sq[t] = (1-mu/(2*L))**t * expct_err_norm_sq[0] + sigma_sq/(mu*L)
        
    return expct_err_norm_sq

n  = 10000    #number of data points
d  = 100     #dimensions
max_iters = 10000

mu_data = np.zeros((n,1))
sigma_data =  np.ones((n,1)) / np.sqrt(1000)
A = random_matrix(mu_data,sigma_data,d)
x_optimal = np.ones((d,1))
x_initial_guess = np.random.normal(0,1,x_optimal.shape)

t = np.arange(max_iters)

x_iters = np.zeros((max_iters,d))
x_iters[0,:] = np.reshape(np.random.normal(0,1,x_optimal.shape),(-1))

#### stochastic_index is denoted by \sigma^2 in the notes.
stochastic_index = np.random.permutation(n)

if(max_iters%n != 0 and int(max_iters/n) != 0):
    stochastic_index = np.concatenate(np.tile(stochastic_index, int(max_iters/n)), stochastic_index[:(max_iters%n)])
elif(max_iters%n == 0):
    stochastic_index = np.tile(stochastic_index, int(max_iters/n))
elif(int(max_iters/n) == 0):
    stochastic_index = stochastic_index[:(max_iters%n)]
else:
    print("Something went wrong.")

L = n * np.max(np.diagonal(np.transpose(A) @ A))
alpha = 1/(2*L)
epsilon_sigma = np.array([0.0,0.01,0.1,1.0])

fig, ax = plt.subplots(1,2, figsize=(10,7.5),dpi=80)
plt.subplots_adjust(hspace = 0.001)
ax[0].set_yscale('log')
ax[0].set_xlabel('iterations')

for i in range(len(epsilon_sigma)):
    epsilon = np.random.normal(0,epsilon_sigma[i], (n,1))
    b = A @ x_optimal + epsilon

    theoretical_err_norm_sq = theoretical_convergence(A,b,x_optimal,x_initial_guess,max_iters,L)
    err_norm_sq = np.zeros((max_iters,1))
    err_norm_sq[0] = np.linalg.norm(x_iters[0,:]-x_optimal)**2.0

    for j in range(1,max_iters):
        x_iters[j,:] = x_iters[j-1,:] - alpha*n*(np.dot(x_iters[j-1,:],A[stochastic_index[j],:])-b[stochastic_index[j]])*A[stochastic_index[j],:]
        
        err_norm_sq[j] = np.linalg.norm(x_iters[j,:]-x_optimal)

    ax[0].plot(t[1:], theoretical_err_norm_sq[1:], label = f'Theo noise \u03C3 = {epsilon_sigma[i]}',linestyle='--')
    ax[0].plot(t[1:], err_norm_sq[1:, 0], label = f'SGD noise \u03C3 = {epsilon_sigma[i]}')
    ax[1].plot(t[1:], err_norm_sq[1:, 0], label = f'SGD noise \u03C3 = {epsilon_sigma[i]}')

ax[0].legend()
ax[1].legend()
legend = plt.legend(loc='best', shadow=False, fontsize='medium')
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('Q7parta.pdf')