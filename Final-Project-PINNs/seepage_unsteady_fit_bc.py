import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

FIGSIZE=(12,8)
np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    """ class for physics informed nn """

    def __init__(self, X, X_left_boundary, X_right_boundary, u, layers, lb, ub, alpha=1, b=0, betas=[0.,0.], optimizer_type="adam"):
        self.lb = lb
        self.ub = ub

        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u
        self.layers = layers
        self.alpha = alpha
        self.alpha_const = tf.constant(self.alpha, dtype=tf.float32, shape=[1,1])
        self.optimizer_type = optimizer_type

        # left and right Neumann conditions
        self.b = b
        self.b_const = tf.constant(self.b, dtype=tf.float32, shape=[1,1])

        self.beta_L = tf.constant(betas[0], dtype=tf.float32, shape=[1,1])
        self.beta_R = tf.constant(betas[1], dtype=tf.float32, shape=[1,1])

        # initialise the NN
        # tf variables
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))


        # initialise parameters
        self.lambda_1 = tf.Variable([np.log(0.05)], dtype=tf.float32) # k = e^{lambda_1}
        # self.k = tf.constant([0.1], dtype=tf.float32)
        # self.lambda_1 = tf.constant([np.log(k)], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32) # neumann BC 

        # define the placeholder variables
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        # define boundary variables 
        self.x_left_boundary = X_left_boundary[:,0:1]
        self.x_right_boundary = X_right_boundary[:,0:1]
        self.t_boundary = X_left_boundary[:,1:2] 
        
        # define boundary placeholders
        self.x_left_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.x_left_boundary.shape[1]])
        self.x_right_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.x_right_boundary.shape[1]])
        self.t_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.t_boundary.shape[1]])

        # NN structure for u and f
        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred = self.net_f(self.x_tf, self.t_tf)

        # boundary residual
        self.f_left_boundary = self.net_left_boundary(self.x_left_boundary_tf, self.t_boundary_tf)
        self.f_right_boundary = self.net_right_boundary(self.x_right_boundary_tf, self.t_boundary_tf) 

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) \
                    + self.alpha_const * tf.reduce_mean(tf.square(self.f_pred)) \
                    + self.beta_L * tf.reduce_mean(tf.square(self.f_left_boundary)) \
                    + self.beta_R * tf.reduce_mean(tf.square(self.f_right_boundary)) 
        
        # define a BFGS optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)


    def initialize_NN(self, layers):
        """
        takes input variable `layers`, a list of layer widths
        initialise the nn and return weights, biases
        lists of tf.Variables corresponding to each layer
        """
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        """ Xavier initialization for the weights """
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


    def neural_net(self, X, weights, biases):
        """ returns the graph for the nn """

        num_layers = len(weights) + 1

        # Rescale the variable X
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1

        # Apply sigma(WX + b) to each layer up until last one
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))

        # last layer, no activation function
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        """ neural network structure for u """
        u = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        lambda_1 = tf.exp(self.lambda_1)
        #source = tf.constant(self.q, dtype=tf.float32, shape=[1,1])
        u = self.net_u(x,t)
        u_x = tf.gradients(u, x)[0]
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        #f = lambda_1 * u_x * u + source
        f = u_t - lambda_1 * (u_x * u_x + u * u_xx)
        # f = u_t - self.k * (u_x * u_x + u * u_xx)
        return f
    
    def net_left_boundary(self, x, t):
        """ left boundary residual """
        u = self.net_u(x,t)
        u_x = tf.gradients(u, x)[0]

        f = u_x - self.lambda_2
        return f

    def net_right_boundary(self, x, t):
        """ right boundary residual """ 
        u = self.net_u(x,t)
        u_x = tf.gradients(u, x)[0]

        f = u_x - self.b_const 
        return f

    def callback(self, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, np.exp(lambda_1), lambda_2))

    def train(self, nIter):
        
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                self.x_left_boundary_tf : self.x_left_boundary,
                self.x_right_boundary_tf : self.x_right_boundary,
                self.t_boundary_tf : self.t_boundary}
        
        start_time = time.time()
        if self.optimizer_type == "adam":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    lambda_2_value = self.sess.run(self.lambda_2) 

                    print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.3f, Time: %.2f' %
                          (it, loss_value, np.exp(lambda_1_value), lambda_2_value, elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                     feed_dict = tf_dict,
                                     fetches = [self.loss, self.lambda_1, self.lambda_2],
                                     loss_callback = self.callback)

        else:
            self.optimizer.minimize(self.sess,
                                     feed_dict = tf_dict,
                                     fetches = [self.loss, self.lambda_1, self.lambda_2],
                                     loss_callback = self.callback)
        
    def predict(self, X_star):
        tf_dict = {self.x_tf: X_star[:,0:1], self.t_tf: X_star[:,1:2], 
                self.x_left_boundary_tf : self.x_left_boundary, 
                self.x_right_boundary_tf : self.x_right_boundary,
                self.t_boundary_tf : self.t_boundary} 

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        f_left = self.sess.run(self.f_left_boundary, tf_dict)
        f_right = self.sess.run(self.f_right_boundary, tf_dict) 

        return u_star, f_star, f_left, f_right

    def save(self, savename):
        """ saves model variables """
        saver = tf.train.Saver()
        savepath = saver.save(self.sess, "saved_models/%s" %(savename))
        print("Model saved in path: %s" %(savepath))


    def load(self, savename):
        saver = tf.train.Saver()
        saver.restore(self.sess, "saved_models/%s" %(savename))


def FD1D_seepage(u_initial,x,t,dx,dt,K,a):
    u_old = u_initial
    u_new = u_initial
    u = []
    u.extend(u_new)
    const = K * dt / (dx * dx)
    for itime in range (1,len(t)):
        for idx in range(1,len(x)-1):
           u_new[idx] = const * ((u_old[idx+1] - u_old[idx])**2 + u_old[idx]*(u_old[idx+1] - 2*u_old[idx] + u_old[idx-1]))+ u_old[idx]
        
        #if boundary on the left hand size
        u_new[0] = u_new[1] - a * dx

        #if boundary on the right hand size
        u_new[len(x)-1] = u_new[len(x)-2]
         
        #update u
        u.extend(u_new)
        u_old = u_new
    u = np.reshape(u,(len(t),len(x)))
    return u

def plot_fd_solution(xx, FD_sol, i, format_str="-"):
    plt.plot(xx, FD_sol[i,:], format_str, linewidth=2)
    plt.ylim([0, np.max(np.max(FD_sol))*1.2])

def plot_model_solution(xx, t, NN, format_str="-"):
    tt = np.ones(xx.shape) * t
    X = np.stack((xx, tt)).T
    u_pred, f_pred, _, _ = NN.predict(X)

    plt.plot(xx, u_pred, format_str, linewidth=2)
    plt.ylim([0, None])

def plot_timestamps(x, dt, FD_sol, model, ind_tests, FD_fit=None):
    plt.rcParams.update({"font.size" : 16})
    for ind in ind_tests:
        plt.figure(figsize=(8,5))
        plot_fd_solution(x, FD_sol, ind, "-k")
        t_ind = ind * dt
        plot_model_solution(x, t_ind, model, "--r")

        if FD_fit is not None:
            plot_fd_solution(x, FD_fit, ind, "-b")

        plt.title("time = %g" %(t_ind))
        plt.xlabel("x")
        plt.ylabel("h")
        plt.grid(True, which="both")

        plt.legend(["PDE Solution with true values", "PINN", "PDE Solution with recovered values"], loc="lower right")
        plt.savefig("unsteady_figures/unsteady_bc_noiseless_res_%d.png" %(ind))
        plt.close()

def plot_animation(x, dt, FD_sol, model, ind_tests, FD_fit=None):
    for ind in ind_tests:
        plt.figure()
        plot_fd_solution(x, FD_sol, ind)
        t_ind = ind * dt
        plot_model_solution(x, t_ind, model)

        if FD_fit is not None:
            plot_fd_solution(x, FD_fit, ind)

        plt.title("time = %g" %(t_ind))
        plt.xlabel("x")
        plt.ylabel("h")
        plt.legend(["numerical", "NN", "numerical with fit K"])
        plt.pause(0.0001)
        plt.clf()
        plt.close()


if __name__ == "__main__":
    load_model = True
    savename = "unsteady_bc_noiseless"


    # NN architecture
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    # PDE problem set up
    n = 50
    t_num = 10000
    t_num_boundary = 100
    
    T_total = 4.0
    L = 1.0
    x = np.linspace(0.0, L, n)
    t = np.linspace(0.0, T_total, t_num)

    dt = t[1] - t[0]
    dx = x[1] - x[0]
    H = 0.1 
    a = 0.1
    b = 0 
    K = 1.0
    
    alpha = 1
    betas = [1, 1]
    
    u_initial = H * np.ones(len(x))
    u0 = u_initial.copy() 
    # u_initial[0] = 0
    
    FD_soloution = FD1D_seepage(u_initial,x,t,dx,dt,K,a)
    
    X, T = np.meshgrid(x,t)
    
    # X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    X_star = np.stack((X.flatten(), T.flatten())).T
    u_star = FD_soloution.flatten()[:,None]
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    
    # points for boundary residual evaluation 
    t_boundary = np.linspace(0, T_total, t_num_boundary)
    # t_boundary = t 
    x_left_boundary = np.zeros(t_boundary.shape)
    x_right_boundary = L * np.ones(t_boundary.shape)
    X_left_boundary = np.stack((x_left_boundary, t_boundary)).T
    X_right_boundary = np.stack((x_right_boundary, t_boundary)).T
    

    add_noise = False
    noise_sd = 0.05 # as a percentage
    N_u = 2000

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    # X_u_train = X_star
    u_train = u_star[idx,:]
    # u_train = u_star



    if add_noise:
        u_train = u_train + noise_sd*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

    if load_model:
        tf.reset_default_graph()
        model = PhysicsInformedNN(X_u_train, X_left_boundary, X_right_boundary, 
                        u_train, layers, lb, ub, b=b, alpha=alpha, betas=betas, optimizer_type="adam") 
        model.load(savename)

        # u_pred, f_pred = model.predict(X_star)
        u_pred, f_pred, f_left, f_right = model.predict(X_u_train)
        k_pred = np.exp(model.sess.run(model.lambda_1))

        FD_sol_fit = FD1D_seepage(u0,x,t,dx,dt,k_pred,a)

        print("k truth: ", K)
        print("k recovered: ", k_pred)
        print("relative error: ", np.abs(K-k_pred)/np.abs(K))

        ind_tests = np.sort(np.random.choice(np.arange(0,t_num), 10, replace=False))
        ind_tests = np.arange(0,t_num, int(t_num/50))

        # plot_animation(x, dt, FD_soloution, model, ind_tests, FD_sol_fit)
        print("k: ", model.sess.run(model.lambda_1))
        print("a: ", model.sess.run(model.lambda_2))

        #### Generate plot of the data #### 

        plt.rcParams.update({"font.size" : 16})
        plt.figure(figsize=(8,4))
        p = plt.pcolor(T, X, FD_soloution, cmap="jet")
        plt.colorbar(p)
        plt.xlabel("t")
        plt.ylabel("x") 
        plt.plot(X_u_train[:,1], X_u_train[:,0], 'xk') 
        plt.savefig("unsteady_training_data.png")
        plt.close() 


        ind_tests = np.arange(0,t_num, int(t_num/3))
        ind_tests = np.append(ind_tests, t_num-1)
        plot_timestamps(x, dt, FD_soloution, model, ind_tests, FD_sol_fit)

    else:
        # Train model
        model = PhysicsInformedNN(X_u_train, X_left_boundary, X_right_boundary, 
                u_train, layers, lb, ub, b=b, alpha=alpha, betas=betas, optimizer_type="adam") 
        model.train(100000)

        # u_pred, f_pred = model.predict(X_star)
        u_pred, f_pred, f_left, f_right = model.predict(X_u_train)
        k_pred = np.exp(model.sess.run(model.lambda_1))

        FD_sol_fit = FD1D_seepage(u0,x,t,dx,dt,k_pred,a)

        print("k truth: ", K)
        print("k recovered: ", k_pred)
        print("relative error: ", np.abs(K-k_pred)/np.abs(K))

        ind_tests = np.sort(np.random.choice(np.arange(0,t_num), 10, replace=False))
        ind_tests = np.arange(0,t_num, int(t_num/50))

        plot_animation(x, dt, FD_soloution, model, ind_tests, FD_sol_fit)
        model.save(savename)
    


