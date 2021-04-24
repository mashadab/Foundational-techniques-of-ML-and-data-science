import sys 
import tensorflow as tf
import matplotlib.pyplot as plt 
import scipy.io 
import numpy as np 
from scipy.interpolate import griddata 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time 
from seepage_pinns_flows import SeepagePINNFlows, SeepageAnalytic


FIGSIZE=(12,8)
plt.rcParams.update({'font.size' : 16})
np.random.seed(1234)
tf.set_random_seed(1234) 

def load_data(n):
    """ loads dataset n"""
    data = scipy.io.loadmat("data/exp%d.mat" %(n)) 
    Q = data['Q'][0][0]
    K_truth = data['K'][0][0]
    
    x_data = data['xexp'][:,0]
    u_data = data['hexp']
    print(x_data)
    L = data['L'][0][0]
    W = data['w'][0][0]
   
    x_data = L - x_data 
    q_data = np.ones(x_data.shape) * Q/W 

    X_data = np.stack((x_data, q_data)).T

    return X_data, u_data, L, W, K_truth 


def load_all():
    """ load all training data into a dictionary 
    stored in order of X, u, L, W, k""" 
    training_data = dict() 
    for i in range(7):
        training_data[i+1] = load_data(i+1) 

    return training_data 

def make_training_set(ind_list, training_data):
    """ compile the training set corresponding
    to experiments listed in ind_list """ 
    
    exp = training_data[ind_list[0]] 
    X_train = exp[0]
    u_train = exp[1] 

    for i in ind_list[1:]: 
        exp = training_data[i]
        X_train = np.append(X_train, exp[0], axis=0)
        u_train = np.append(u_train, exp[1], axis=0)

    return X_train, u_train 

training_data = load_all() 
training_list = [4,5,6,7] 
X_train, u_train = make_training_set(training_list, training_data) 
X, u, L, W, k = training_data[1] 

alpha = 1000000

# Adding collocation points
Q_colloc_list = np.linspace(1e-5, 1e-4, 10)
x_colloc = np.array([]) 
q_colloc = np.array([])
n_colloc = 20

for Q in Q_colloc_list: 
    x_locs = np.linspace(0, L, n_colloc)
    q_locs = np.ones(x_locs.shape) * Q/W 
    x_colloc = np.append(x_colloc, x_locs) 
    q_colloc = np.append(q_colloc, q_locs) 

X_colloc = np.stack((x_colloc, q_colloc)).T 

# define a model 
layers = [2, 20, 20, 20, 20, 20, 20, 20, 1]
model = SeepagePINNFlows(X_train, u_train, layers, 0, L, k, X_colloc=X_colloc, alpha=1e6, optimizer_type="both")
model.train(50000)
model.save("steady_exp_flow_large")

model_small = SeepagePINNFlows(X_train, u_train, layers, 0, L, k, X_colloc=X_colloc, alpha=100, optimizer_type="both")
model_small.train(50000)
model_small.save("steady_exp_flow_small")

model_none = SeepagePINNFlows(X_train, u_train, layers, 0, L, k, X_colloc=None, alpha=0, optimizer_type="both")
model_none.train(50000)
model_none.save("steady_exp_flow_none")

for i in range(1,8): 
    plt.figure(figsize=(8,5))
    X, u, L, W, k = training_data[i] 
    u_pred, f_pred = model.predict(X) 
    plt.plot(L-X[:,0], u, 'ok')
    plt.plot(L-X[:,0], u_pred, '^b') 
    u_pred, f_pred = model_small.predict(X) 
    plt.plot(L-X[:,0], u_pred, 'xg') 
    u_pred, f_pred = model_none.predict(X) 
    plt.plot(L-X[:,0], u_pred, 'dr') 
    plt.xlabel("x")
    plt.ylabel("h")
    plt.grid(True, which="both")
    if i in training_list:
        plt.title("Set %d (training), q = %.3e" %(i, X[0,1])) 
    else: 
        plt.title("Set %d (test), q = %.3e" %(i, X[0,1])) 
    plt.legend(["Data", "$\\alpha = 10^6$", "$\\alpha = 10^2$", "$\\alpha = 0$"], loc="lower right")
    plt.savefig("steady_figures/flow_exp_%d" %(i))
    plt.show()
    plt.close()




for i in range(1,8): 
    plt.figure()
    X, u, L, W, k = training_data[i] 
    u_pred, f_pred = model_none.predict(X) 
    plt.plot(L-X[:,0], u, 'or')
    plt.plot(L-X[:,0], u_pred, 'ob') 
    plt.xlabel("x")
    plt.ylabel("h")
    if i in training_list:
        plt.title("Set %d (training), q = %.3e" %(i, X[0,1])) 
    else: 
        plt.title("Set %d (test), q = %.3e" %(i, X[0,1])) 
    plt.show()




for i in range(1,8): 
    plt.figure()
    X, u, L, W, k = training_data[i] 
    u_pred, f_pred = model.predict(X) 
    plt.semilogy(L-X[:,0], np.abs(f_pred), 'ob') 
    plt.xlabel("x")
    plt.ylabel("h")
    if i in training_list:
        plt.title("Set %d (training), q = %.3e" %(i, X[0,1])) 
    else: 
        plt.title("Set %d (test), q = %.3e" %(i, X[0,1])) 
    plt.show()
