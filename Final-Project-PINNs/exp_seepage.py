import sys 
import tensorflow as tf
import matplotlib.pyplot as plt 
import scipy.io 
import numpy as np 
from scipy.interpolate import griddata 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time 
from seepage import SeepageSimplePINN, SeepageSteadyWithBCsPINN, seepage

FIGSIZE=(12,8)
plt.rcParams.update({'font.size' : 16})
np.random.seed(1234)
tf.set_random_seed(1234) 


# load experimental data 
dataset_no = 1
data = scipy.io.loadmat("data/exp%d.mat" %(dataset_no))
Q = data['Q'][0][0]
K_truth = data['K'][0][0]

x_data = data['xexp']
u_data = data['hexp']
L = data['L'][0][0]
W = data['w'][0][0]

X_u_train = x_data
u_train = u_data
lb = np.min(x_data)
ub = np.max(x_data)

# NN architecture
layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# layers = [1, 20, 20, 20, 20, 20, 1]

q = Q/W
alpha = 1/q
n_colloc = 100
X_colloc = np.linspace(0,L,n_colloc).reshape([n_colloc, 1])

x_left = np.array([[0]])
x_right = np.array([[L]]) 
x_bcs = [x_left, x_right]
bcs = [0, Q/W]
betas = [0, 1/q] 
lambda_guess = np.log(0.1)

model = SeepageSteadyWithBCsPINN(X_u_train, u_train, x_bcs, bcs, layers, lb, ub,
        X_colloc=X_colloc, lambda_guess=lambda_guess, alpha=alpha, betas=betas, optimizer_type="both")

# model = SeepageSimplePINN(X_u_train, u_train, layers, lb, ub, -Q/W)
model.train(20000)

# recovered k value 
K_pred = np.exp(model.sess.run(model.lambda_1))
print("k truth: ", K_truth) 
print("k recovered: ", K_pred)
print("relative error: ", np.abs(K_truth-K_pred)/np.abs(K_truth))

# predictions
u_pred = model.predict(X_u_train) 

 # checking the solution using the inferred porosity with analytic model 
L = 1.65 
x = np.linspace(0, L, 100) 
hs = u_data[0] 
u_analytic = np.flipud(seepage(x, L, W, Q, hs, K_truth))
u_analytic_pred = np.flipud(seepage(x, L, W, Q, hs, K_pred))
u_eval_model = model.predict(x.reshape([x.shape[0], 1]))

plt.rcParams.update({"font.size" : 16}) 
plt.figure(figsize=(8,5))
plt.plot(X_u_train, u_train, 'ok')
plt.plot(X_u_train, u_pred[0], 'xb') 
plt.plot(x, u_analytic, '--k')
plt.plot(x, u_analytic_pred, '-b')
plt.xlabel("x")
plt.ylabel("h")
plt.grid(True, which="both")
plt.title("u (training data set %d)" %(dataset_no)) 
plt.legend(["Data", "NN", "PDE solution with measured K", "PDE solution with recovered K"])
plt.savefig("steady_figures/steady_k_exp_dataset%d.png" %(dataset_no))
plt.show()

"""
# plot data vs trained predictions
plt.figure(figsize=FIGSIZE)
plt.subplot(111)
plt.plot(X_u_train, u_train, 'or')
plt.plot(X_u_train, u_pred[0], 'ob') 
plt.plot(x, u_analytic, '--k')
plt.plot(x, u_analytic_pred, '-b')
plt.xlabel("x")
plt.ylabel("h")
plt.grid(True, which="both")
plt.title("u (training data set %d)" %(dataset_no)) 
plt.legend(["Data", "Prediction", "Analytic with measured K", "Analytic with fitted K"])
plt.savefig("predictions_dataset%d.png" %(dataset_no))

# plot model vs analytic
plt.figure(figsize=FIGSIZE)
plt.subplot(111)
plt.plot(x, u_eval_model[0], 'r')
plt.plot(x, u_analytic, '--k')
plt.plot(x, u_analytic_pred, '--b')
plt.xlabel("x")
plt.ylabel("h")
plt.grid(True, which="both")
plt.title("NN vs analytic (training data set %d)" %(dataset_no)) 
plt.legend(["Prediction", "Analytic with measured K", "Analytic with fitted K"])
plt.savefig("nn_analytic_dataset_%d.png" %(dataset_no))

plt.show() 
"""
