#class   : Foundations of ML and Data Science
#homework: 3
#question: 7
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#Stochastic Gradient Descent
import numpy as np
import matplotlib.pyplot as plt
from hw3q7SGD_functions import generate_random_matrix, theoretical_conv, stochastic_index_compute
plt.rcParams["font.family"] = "Serif"

# Part (a)
n  = 10000        #number of data points
d  = 100          #dimensions
max_iters = 10000 #maximum iteration
mu_data = np.zeros((n,1)) #vector of means
sigma_data =  np.ones((n,1)) / np.sqrt(1000) #vector of std dev
A = generate_random_matrix(mu_data,sigma_data,d) #Random matrix n X d
x_opt = np.ones((d,1)) #optimal value of x
x_init_guess = np.random.normal(0,1,x_opt.shape) #initial guess
t = np.arange(max_iters)

x_iters = np.zeros((max_iters,d))
x_iters[0,:] = np.reshape(np.random.normal(0,1,x_opt.shape),(-1))

stochastic_index = stochastic_index_compute(n,max_iters) #### sigma^2 from notes stochastic_index

L = n * np.max(np.diagonal(np.transpose(A) @ A))
alpha = 1/(2*L)
epsilon_sigma = np.array([0.0,0.01,0.1,1.0])

fig, ax = plt.subplots(1,2, figsize=(10,7.5),dpi=80)
plt.subplots_adjust(hspace = 0.001)
ax[0].set_yscale('log')
ax[0].set_xlabel('Iterations')
ax[1].set_xlabel('Iterations')
ax[0].set_ylabel('Mean Squared Error Norm')
for i in range(len(epsilon_sigma)):
    epsilon = np.random.normal(0,epsilon_sigma[i], (n,1))
    b = A @ x_opt + epsilon

    theoretical_err_norm_sq = theoretical_conv(A,b,x_opt,x_init_guess,max_iters,L)
    err_norm_sq = np.zeros((max_iters,1))
    err_norm_sq[0] = np.linalg.norm(x_iters[0,:]-x_opt)**2.0

    for j in range(1,max_iters):
        x_iters[j,:] = x_iters[j-1,:] - alpha*n*(np.dot(x_iters[j-1,:],A[stochastic_index[j],:])-b[stochastic_index[j]])*A[stochastic_index[j],:]
        
        err_norm_sq[j] = np.linalg.norm(x_iters[j,:]-x_opt)

    ax[0].plot(t[1:], err_norm_sq[1:, 0], label = f'SGD noise \u03C3 = {epsilon_sigma[i]}')
    ax[1].plot(t[1:], err_norm_sq[1:, 0], label = f'SGD noise \u03C3 = {epsilon_sigma[i]}')
    ax[0].plot(t[1:], theoretical_err_norm_sq[1:], label = f'Theo noise \u03C3 = {epsilon_sigma[i]}',linestyle='--')

ax[0].legend(loc='best', shadow=False, fontsize='medium')
ax[1].legend(loc='best', shadow=False, fontsize='medium')
manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('Q7parta.pdf')


#Part (b)
sigma_data = np.zeros((n))
for j in range(n):
    sigma_data[j] = 1/np.sqrt((j+1)*1000)

A = generate_random_matrix(mu_data,sigma_data,d)
x_iters = np.zeros((max_iters,d))
x_iters[0,:] = np.reshape(np.random.normal(0,1,x_opt.shape), (-1))

stochastic_index = stochastic_index_compute(n,max_iters)
    
fig, ax = plt.subplots(1,2, figsize=(10,10))
plt.subplots_adjust(hspace = 0.001)
ax[0].set_yscale('log')
ax[0].set_xlabel('Iterations')
ax[1].set_xlabel('Iterations')
ax[0].set_ylabel('Mean Squared Error Norm')

for i in range(len(epsilon_sigma)): # Looping over each epsilon
    epsilon = np.random.normal(0, epsilon_sigma[i], (n,1))
    b = A @ x_opt + epsilon
    theoretical_err_norm_sq = theoretical_conv(A, b, x_opt, x_init_guess, max_iters,L)     # Theoretical rate for each epsilon
    err_norm_sq = np.zeros((max_iters,1))
    err_norm_sq[0] = np.linalg.norm(x_iters[0,:]-x_opt)**2
    ### SGD for each epsilon
    for j in range(1, max_iters):
        x_iters[j,:] = x_iters[j-1,:] - alpha*n*(np.dot(x_iters[j-1,:],A[stochastic_index[j],:])-b[stochastic_index[j]])*A[stochastic_index[j],:]
        err_norm_sq[j] = np.linalg.norm(x_iters[j,:]-x_opt)

    ax[0].plot(t[1:], err_norm_sq[1:, 0], label = f'SGD noise \u03C3 = {epsilon_sigma[i]}')
    ax[1].plot(t[1:], err_norm_sq[1:, 0], label = f'SGD noise \u03C3 = {epsilon_sigma[i]}')
    ax[0].plot(t[1:], theoretical_err_norm_sq[1:], label = f'Theo noise \u03C3 = {epsilon_sigma[i]}',linestyle='--')

ax[0].legend(loc='best', shadow=False, fontsize='medium')
ax[1].legend(loc='best', shadow=False, fontsize='medium')

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.savefig('Q7partb.pdf')