import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class SeepagePINNFlows:
    # Initialize the class
    def __init__(self, X, u, layers, lb, ub, kappa, X_colloc=None, alpha=1, optimizer_type="adam"):
        
        self.lb = lb # lower bound for x 
        self.ub = ub # upper bound for x
        self.kappa = kappa # permeability
        self.layers = layers # architecture 
        self.alpha = alpha  # regularization parameter
        self.X_colloc = X_colloc # Collocation points
        self.optimizer_type = optimizer_type

        # Define the regularization parameter as tf.constant
        self.alpha_const = tf.constant(self.alpha, dtype=tf.float32, shape=[1,1])
        
        self.x = X[:,0:1]
        self.q = X[:,1:2]
        self.u = u
        
        print(self.x.shape[1])
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # inputs and outputs
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.q_tf = tf.placeholder(tf.float32, shape=[None, self.q.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred = self.net_u(self.x_tf, self.q_tf)
        self.f_pred = self.net_f(self.x_tf, self.q_tf)
        
        if X_colloc is None:
            # simple
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                        self.alpha_const * tf.reduce_mean(tf.square(self.f_pred))
        else:
            # Using collocation points for PDE residual as well  
            self.x_colloc = X_colloc[:, 0:1] # x points
            self.q_colloc = X_colloc[:, 1:2] # q points 
            self.x_colloc_tf = tf.placeholder(tf.float32, shape=[None, self.x_colloc.shape[1]]) 
            self.q_colloc_tf = tf.placeholder(tf.float32, shape=[None, self.q_colloc.shape[1]])

            # PDE residual at collaction points 
            self.f_colloc = self.net_f(self.x_colloc_tf, self.q_colloc_tf) 

            # Loss to account for training data on the state, PDE residual of data, and PDE resdiaul
            # at collocation points
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) \
                        + self.alpha_const * tf.reduce_mean(tf.square(self.f_pred)) \
                        + self.alpha_const * tf.reduce_mean(tf.square(self.f_colloc))
        
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 80,
                                                                           'maxls': 80,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
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
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_u(self, x, q):  
        u = self.neural_net(tf.concat([x,q],1), self.weights, self.biases)
        return u
    
    def net_f(self, x, q):
        u = self.net_u(x,q)
        u_x = tf.gradients(u, x)[0]
        kappa = tf.constant(self.kappa, dtype=tf.float32, shape=[1,1])
        
        f = kappa * u * u_x + q 
        
        return f
    
    def callback(self, loss):
        print('Loss: %e' %(loss)) 
        
    def train(self, nIter):
        if self.X_colloc is None:
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u}
               
        else: 
            tf_dict = {self.x_tf: self.x, self.q_tf: self.q, self.u_tf: self.u, 
                    self.x_colloc_tf: self.x_colloc, self.q_colloc_tf: self.q_colloc}
        
        start_time = time.time()
        if self.optimizer_type == "adam":
            # use adam only
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' % 
                          (it, loss_value, elapsed))
                    start_time = time.time()

        elif self.optimizer_type == "both":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    print('It: %d, Loss: %.3e, Time: %.2f' % 
                          (it, loss_value, elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)

        else:
            # use BFGS 
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss],
                                    loss_callback = self.callback)
        
        
    def predict(self, X_star):
        
        tf_dict = {self.x_tf: X_star[:,0:1], self.q_tf: X_star[:,1:2]}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star

    def save(self, savename):
        """ saves model variables """
        saver = tf.train.Saver()
        savepath = saver.save(self.sess, "saved_models/%s" %(savename))
        print("Model saved in path: %s" %(savepath))


    def load(self, savename):
        saver = tf.train.Saver()
        saver.restore(self.sess, "saved_models/%s" %(savename))


class SeepageAnalytic:
    def __init__(self, L, W, hs, k):
        self.L = L
        self.W = W
        self.hs = hs 
        self.k = k

    def eval(self, xx, Q):
        return np.sqrt(self.hs**2 + 2 * Q * (self.L - xx)/(self.W*self.k))

class SeepagePerturbed:
    def __init__(self, L, W, hs, k):
        self.L = L
        self.W = W
        self.hs = hs 
        self.k = k

    def eval(self, xx, Q):
        return np.sqrt(self.hs**2 + 2 * Q * (self.L - xx)/(self.W*self.k)) - 0.5 * np.tanh(self.L - xx)


def test_model(n_test, Q, NN_model, analytic_model):
    """ test the model for a given flow """ 
    plot_model_comparison(n_test, Q, NN_model, analytic_model) 
    plot_pde_residual(n_test, Q, NN_model, analytic_model) 


def compare_colloc(Q, NN_model_colloc, NN_model_none, analytic_model):

    # x_locs = np.random.uniform(high=analytic_model.L, size=n_test)
    plt.rcParams.update({"font.size" : 16})
    x_locs = np.linspace(0, analytic_model.L, 100)
    q_locs = np.ones(x_locs.shape) * Q / analytic_model.W 
    u_locs = analyticModel.eval(x_locs, Q) 
    X = np.stack((x_locs, q_locs)).T 

    u_colloc, f_colloc = NN_model_colloc.predict(X) 
    u_none, f_none = NN_model_none.predict(X) 
    x_locs = analytic_model.L - x_locs

    plt.figure(figsize=(8,5))
    plt.plot(x_locs, u_locs, '-k', linewidth=2) 
    plt.plot(x_locs, u_none, '--r', linewidth=2)
    plt.plot(x_locs, u_colloc, '--b', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("h")
    plt.grid(True, which="both")
    plt.title("Q = %g" %(Q)) 
    plt.legend(["True solution", "NN without collocation", "NN with collocation"], loc="best")

    
def plot_model_comparison(n_test, Q, NN_model, analytic_model): 
    """ plot the neural network model with analytic model """
    x_locs = np.random.uniform(high=analytic_model.L, size=n_test)
    q_locs = np.ones(x_locs.shape) * Q / analytic_model.W 
    u_locs = analyticModel.eval(x_locs, Q) 

    X = np.stack((x_locs, q_locs)).T 
    u_pred, f_pred = NN_model.predict(X) 

    plt.figure()
    plt.subplot(111)
    plt.plot(x_locs, u_locs, 'or')
    plt.plot(x_locs, u_pred, 'ob')
    plt.xlabel("x")
    plt.ylabel("h")
    plt.title("State: Q = %g" %(Q))
    plt.legend(["Analytic", "NN"]) 
    plt.show()

def plot_pde_residual(n_test, Q, NN_model, analytic_model): 
    """ plot the pde residual from NN model """
    x_locs = np.linspace(0, analytic_model.L, n_test)  
    q_locs = np.ones(x_locs.shape) * Q / analytic_model.W 
    u_locs = analyticModel.eval(x_locs, Q) 

    X = np.stack((x_locs, q_locs)).T 
    u_pred, f_pred = NN_model.predict(X) 

    plt.figure()
    plt.subplot(111)
    plt.plot(x_locs, f_pred, 'ob')
    plt.xlabel("x")
    plt.ylabel("h")
    plt.title("PDE residual: Q = %g" %(Q))
    plt.legend(["Analytic", "NN"]) 
    plt.show()

def compare_perturbed():
    exact = SeepageAnalytic(1, 1, 0, 0.1)
    perturbed = SeepagePerturbed(1, 1, 0, 0.1) 
    x = np.linspace(0, 1, 100) 
    plt.figure()
    plt.plot(x, exact.eval(x, 1), '--r')
    plt.plot(x, perturbed.eval(x, 1), '-b')
    plt.show()

if __name__ == "__main__":

    # NN architecture
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    alpha = 1.0

    # PDE problem set up     
    L = 1 
    k = 0.1
    hs = 0 
    W = 1

    analyticModel = SeepageAnalytic(L, W, hs, k) 
    # analyticModel = SeepagePerturbed(L, W, hs, k) 

    # data set up 
    n_data = 30
    
    # list of flows for training data
    # Q_list = [0.1, 0.5, 1.0, 2.0, 5.0] 
    Q_list = [1.0, 2.0]

    # range of x values for training data - can change if you want to concentrate
    # the data at boundaries, similar to only providing the BCs. If so, set the 
    # collocation points on the interior to train the model using the PDE

    x_low = 0
    x_high = L
    x_data = np.array([]) 
    q_data = np.array([])
    u_data = np.array([])
     
    # Use analytical solution as training data
    for Q in Q_list: 
        x_locs = np.random.uniform(low=x_low, high=x_high, size=n_data)
        q_locs = np.ones(x_locs.shape) * Q/W 
        u_locs = analyticModel.eval(x_locs, Q) 
       
        x_data = np.append(x_data, x_locs) 
        q_data = np.append(q_data, q_locs) 
        u_data = np.append(u_data, u_locs) 

    u_train = u_data.reshape([u_data.shape[0], 1])
    X_u_train = np.stack((x_data, q_data)).T

    # Adding collocation points - evaluate PDE residual
    Q_colloc_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    x_colloc = np.array([]) 
    q_colloc = np.array([])

    n_colloc = 20 # number of collocation points in x
    for Q in Q_colloc_list: 
        x_locs = np.linspace(0, L, n_colloc) 
        q_locs = np.ones(x_locs.shape) * Q/W 
        x_colloc = np.append(x_colloc, x_locs) 
        q_colloc = np.append(q_colloc, q_locs) 

    X_colloc = np.stack((x_colloc, q_colloc)).T 

    # Add noise to training data if necessary 
    add_noise = False
    noise_sd = 0.05 # as a percentage of maximum value 
    
    if add_noise:
        u_max = np.max(u_train)
        u_train += noise_sd * np.random.randn(u_train.shape[0], 1) * u_max

    lb = 0 
    ub = L
     
    model = SeepagePINNFlows(X_u_train, u_train, layers, lb, ub, k, alpha=alpha, optimizer_type="both", X_colloc=X_colloc) 
    model.train(200000)
    
    model_none = SeepagePINNFlows(X_u_train, u_train, layers, lb, ub, k, alpha=alpha, optimizer_type="both")
    model_none.train(200000)

    # prediction for training data
    u_pred_train, f_pred_train = model.predict(X_u_train) 
    
    # plt.figure()
    # plt.plot(u_train, u_pred_train, 'ob')
    # plt.xlabel("Data")
    # plt.ylabel("Prediction")
    # plt.show()
    
    qq = np.arange(0.5, 10.5, 0.5) 
    for q in qq:
        compare_colloc(q, model, model_none, analyticModel)
        plt.savefig("steady_figures/colloc_%g.png" %(q))
        plt.close()
    
    # test the model
    # test_model(100, 1, model, analyticModel)
    # test_model(100, 2, model, analyticModel)
    # test_model(100, 3, model, analyticModel)
    # test_model(100, 4, model, analyticModel)
    # test_model(100, 5, model, analyticModel)
    # test_model(100, 8, model, analyticModel)
