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

class SeepageSimplePINN:
    """ class for physics informed nn """

    def __init__(self, X, u, layers, lb, ub, q):
        self.lb = lb
        self.ub = ub 

        self.x = X[:, 0:1] 
        self.u = u 
        self.layers = layers

        self.q = q

        # initialise the NN 

        # tf variables
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # initialise parameters 
        self.lambda_1 = tf.Variable([1.0], dtype=tf.float32)

        # define the placeholder variables 
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]]) 

        # NN structure for u and f 
        self.u_pred = self.net_u(self.x_tf) 
        self.f_pred = self.net_f(self.x_tf) 

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))
        
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

    def net_u(self, x):
        """ neural network structure for u """ 
        u = self.neural_net(x, self.weights, self.biases) 
        return u 

    def net_f(self, x):
        lambda_1 = tf.exp(self.lambda_1)
        source = tf.constant(self.q, dtype=tf.float32, shape=[1,1])
        u = self.net_u(x) 
        u_x = tf.gradients(u, x)[0]
        f = lambda_1 * u_x * u + source
        # u_xx = tf.gradients(u_x, x)[0]
        # f = lambda_1 * u_x * u_x + lambda_1 * u_xx * u 
        return f 

    def callback(self, loss, lambda_1):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, rnp.exp(lambda_1)))

    def train(self, nIter):
        tf_dict = {self.x_tf : self.x, self.u_tf : self.u} 
        start_time = time.time() 

        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict) 

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Time: %.2f' % 
                      (it, loss_value, np.exp(lambda_1_value), elapsed))
                start_time = time.time()

        #  self.optimizer.minimize(self.sess,
        #                          feed_dict = tf_dict,
        #                          fetches = [self.loss, self.lambda_1, self.lambda_2],
        #                          loss_callback = self.callback)
        
    def predict(self, X_star):
        tf_dict = {self.x_tf : X_star[:, 0:1]} 
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star


class SeepageSteadyWithBCsPINN:
    """ class for physics informed nn """

    def __init__(self, X, u, x_boundaries, bcs, layers, lb, ub, 
            X_colloc=None, alpha=1, betas=[1,1], lambda_guess=1.0, optimizer_type="adam"):
        """ 
        PINN for steady state seepage equation with the option of neumann BCs and collocation points
    
        """
        # store data
        self.lb = lb
        self.ub = ub 
        self.x = X[:, 0:1] 
        self.u = u 
        self.layers = layers
        self.optimizer_type = optimizer_type

        # weighting for PDE residual
        self.alpha = tf.constant(alpha, dtype=tf.float32, shape=[1,1])

        # points for neumann boundary conditions 
        self.x_left = x_boundaries[0]
        self.x_right = x_boundaries[1]

        # interior collocation points 
        self.x_colloc = X_colloc

        # values of neumann boundary conditions 
        self.a = tf.constant(bcs[0], dtype=tf.float32, shape=[1,1]) 
        self.b = tf.constant(bcs[1], dtype=tf.float32, shape=[1,1])

        # weighting for neumann boundary condition residual. set to 0 for none
        self.beta_left = betas[0] 
        self.beta_right = betas[1]


        # tf variables
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # initialise parameters 
        self.lambda_1 = tf.Variable([lambda_guess], dtype=tf.float32)

        # define the placeholder variables 
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]]) 


        # boundary point evaluations 
        self.x_left_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_right_tf = tf.placeholder(tf.float32, shape=[None, 1]) 

        # NN structure for u and f 
        self.u_pred = self.net_u(self.x_tf) 
        self.f_pred = self.net_f(self.x_tf) 

        # residuals for left and right boundaries 
        self.f_left = self.net_f_left_boundary(self.x_left_tf) 
        self.f_right = self.net_f_right_boundary(self.x_right_tf) 


        print("alpha: ", self.alpha)
        print("left: ", self.a)
        print("right: ", self.b)

        if X_colloc is None:
            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) \
                        + self.alpha * tf.reduce_mean(tf.square(self.f_pred)) \
                        + self.beta_left * tf.reduce_mean(tf.square(self.f_left)) \
                        + self.beta_right * tf.reduce_mean(tf.square(self.f_right))
        else: 
            # interior collocation points 
            self.x_colloc_tf = tf.placeholder(tf.float32, shape=[None, 1]) 
            self.f_colloc = self.net_f(self.x_colloc_tf) 

            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) \
                        + self.alpha * tf.reduce_mean(tf.square(self.f_pred)) \
                        + self.beta_left * tf.reduce_mean(tf.square(self.f_left)) \
                        + self.beta_right * tf.reduce_mean(tf.square(self.f_right)) \
                        + self.alpha * tf.reduce_mean(tf.square(self.f_colloc)) 


        
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

    def net_u(self, x):
        """ neural network structure for u """ 
        u = self.neural_net(x, self.weights, self.biases) 
        return u 

    def net_f(self, x):
        """ neural network structure for PDE residual """ 
        k = tf.exp(self.lambda_1)
        # source = tf.constant(self.q, dtype=tf.float32, shape=[1,1])
        u = self.net_u(x) 
        u_x = tf.gradients(u, x)[0]
        # f = lambda_1 * u_x * u + source
        u_xx = tf.gradients(u_x, x)[0]
        f = k * u_x * u_x + k * u_xx * u 
        # f = u_x * u_x + u_xx * u 
        return f 

    def net_f_left_boundary(self, x):
        """ neural network structure for left boundary residual """
        k = tf.exp(self.lambda_1)
        u = self.net_u(x)
        u_x = tf.gradients(u, x)[0] 
        f = k * u*u_x - self.a 
        return f 

    def net_f_right_boundary(self, x): 
        """ neural network structure for right boundary residual """
        k = tf.exp(self.lambda_1)
        u = self.net_u(x) 
        u_x = tf.gradients(u, x)[0] 
        f = k * u*u_x - self.b
        return f 


    def callback(self, loss, lambda_1):
        print('Loss: %e, l1: %.5f' % (loss, np.exp(lambda_1)))


    def train(self, nIter):
        if self.x_colloc is None:
            print("No collocation points")
            tf_dict = {self.x_tf : self.x, self.u_tf : self.u, 
                self.x_left_tf : self.x_left, self.x_right_tf: self.x_right}
        else:
            print("Using collocation points")
            tf_dict = {self.x_tf : self.x, self.u_tf : self.u, 
                self.x_left_tf : self.x_left, self.x_right_tf: self.x_right,
                self.x_colloc_tf: self.x_colloc}

        start_time = time.time() 
        print(self.x_left, self.x_right, self.sess.run(self.a), self.sess.run(self.b))

        if self.optimizer_type == "adam":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict) 

                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    print('It: %d, Loss: %.3e, Lambda_1: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), elapsed))
                    start_time = time.time()

        elif self.optimizer_type == "both":
            for it in range(nIter):
                self.sess.run(self.train_op_Adam, tf_dict) 

                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value = self.sess.run(self.loss, tf_dict)
                    lambda_1_value = self.sess.run(self.lambda_1)
                    print('It: %d, Loss: %.3e, Lambda_1: %.3f, Time: %.2f' % 
                          (it, loss_value, np.exp(lambda_1_value), elapsed))
                    start_time = time.time()

            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1],
                                    loss_callback = self.callback)

        else: 
            self.optimizer.minimize(self.sess,
                                    feed_dict = tf_dict,
                                    fetches = [self.loss, self.lambda_1],
                                    loss_callback = self.callback)
        
    def predict(self, X_star):
        tf_dict = {self.x_tf : X_star[:, 0:1]} 
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star

    def predictbc(self):
        tf_dict = {self.x_tf : self.x, self.u_tf : self.u, 
                self.x_left_tf : self.x_left, self.x_right_tf: self.x_right}
        f_left = self.sess.run(self.f_left, tf_dict)
        f_right = self.sess.run(self.f_right, tf_dict) 
        return f_left, f_right
        
    def save(self, savename):
        """ saves model variables """
        saver = tf.train.Saver()
        savepath = saver.save(self.sess, "saved_models/%s" %(savename))
        print("Model saved in path: %s" %(savepath))


    def load(self, savename):
        saver = tf.train.Saver()
        saver.restore(self.sess, "saved_models/%s" %(savename))


def seepage(x, L, W, Q, hs, k):
    return np.sqrt(hs**2 + 2 * Q * (L - x)/(W*k))



class SeepageAnalytic:
    def __init__(self, L, W, hs, k, Q):
        self.L = L
        self.W = W
        self.hs = hs 
        self.k = k
        self.Q = Q 

    def eval_unshifted(self, xx):
        return np.sqrt(self.hs**2 + 2 * self.Q * (self.L - xx)/(self.W*self.k))

    def make_training_data(self, n_data):
        x = np.random.uniform(high=self.L, size=n_data).reshape([n_data, 1]) 
        u_train = self.eval_unshifted(x) 
        X_train = self.L - x 
        return X_train, u_train

    def make_training_data(self, n_data):
        x = np.linspace(0, self.L, n_data).reshape([n_data, 1]) 
        u_train = self.eval_unshifted(x) 
        X_train = self.L - x 
        return X_train, u_train

if __name__ == "__main__":

    # NN architecture
    layers = [1, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    # PDE problem set up     
    L = 1
    Q = 0.0001
    k = 0.1
    hs = 0 
    W = 1

    # data set up 
    n_data = 30
    n_test = 1000
    lambda_guess = np.log(1)

    lb = 0 
    ub = 1 

    q = Q/W
    alpha0 = 1
    alpha1 = 1/q
    alpha2 = 1/q**2
    

    flip_domain = True
    add_noise = True
    noise_sd = 0.05 # as a percentage


    x_left = np.array([[0]])
    x_right = np.array([[L]]) 
    x_bcs = [x_left, x_right] 
    bcs = [-Q/W, Q/W] 

    if flip_domain: 
        analytic = SeepageAnalytic(L, W, hs, k ,Q) 
        X_u_train, u_train = analytic.make_training_data(n_data) 
        betas0 = [0, 1]
        betas1 = [0.0/q, 1.0/q]
        betas2 = [0.0/q, 1.0/q**2]

        plt.figure()
        plt.plot(X_u_train, u_train, 'o')
        plt.show

    else:
        X_u_train = np.random.uniform(high=L, size=n_data).reshape([n_data, 1])
        u_train = seepage(X_u_train, L, W, Q, hs, k)
        betas = [1.0, 0.0]

        plt.figure()
        plt.plot(X_u_train, u_train, 'o')
        plt.show

    if add_noise:
        u_max = np.max(u_train)
        u_train += noise_sd * np.random.randn(u_train.shape[0], 1) * u_max

    model = SeepageSteadyWithBCsPINN(X_u_train, u_train, x_bcs, bcs, layers, lb, ub, lambda_guess=lambda_guess, alpha=alpha0, betas=betas0, optimizer_type="both")
    model.train(100000)
    # model.save("steady_k_noscale")

    model_lin = SeepageSteadyWithBCsPINN(X_u_train, u_train, x_bcs, bcs, layers, lb, ub, lambda_guess=lambda_guess, alpha=alpha1, betas=betas1, optimizer_type="both")
    model_lin.train(100000)
    # model_lin.save("steady_k_linscale")

    model_quad = SeepageSteadyWithBCsPINN(X_u_train, u_train, x_bcs, bcs, layers, lb, ub, lambda_guess=lambda_guess, alpha=alpha2, betas=betas2, optimizer_type="both")
    model_quad.train(100000)
    # model_quad.save("steady_k_quadscale")

    # model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub, Q/W)
    
    # recovered k value 
    print("k truth: ", k) 
    k_pred = np.exp(model.sess.run(model.lambda_1))
    analytic_none = SeepageAnalytic(L, W, hs, k_pred, Q) 
    print("k recovered no scaling: ", k_pred)
    print("relative error: ", np.abs(k-k_pred)/np.abs(k))

    k_pred = np.exp(model_lin.sess.run(model_lin.lambda_1))
    analytic_lin = SeepageAnalytic(L, W, hs, k_pred, Q) 
    print("k recovered linear scaling: ", k_pred)
    print("relative error: ", np.abs(k-k_pred)/np.abs(k))

    k_pred = np.exp(model_quad.sess.run(model_quad.lambda_1))
    analytic_quad = SeepageAnalytic(L, W, hs, k_pred, Q) 
    print("k recovered quadratic scaling: ", k_pred)
    print("relative error: ", np.abs(k-k_pred)/np.abs(k))

    x = np.linspace(0, L, 200) 
    u_exact = analytic.eval_unshifted(x)  
    u_none = analytic_none.eval_unshifted(x) 
    u_lin = analytic_lin.eval_unshifted(x)
    u_quad = analytic_quad.eval_unshifted(x)

    X_test = x.reshape([x.shape[0], 1]) 

    plt.rcParams.update({"font.size" : 16})
    plt.figure(figsize=(8,5)) 
    plt.plot(X_u_train, u_train, 'ok') 
    # plot the 0th case 
    u_pred, f_pred = model.predict(X_test) 
    plt.plot(x, u_pred[:,0], '--r', linewidth=2)
    u_pred, f_pred = model_lin.predict(X_test) 
    plt.plot(x, u_pred[:,0], '--g', linewidth=2)
    u_pred, f_pred = model_quad.predict(X_test) 
    plt.plot(x, u_pred[:,0], '--b', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("h") 
    plt.grid(True, which="both")
    plt.legend(["Data", "$\\alpha = \\beta = 1 $", "$\\alpha = \\beta = W/Q$", "$\\alpha = \\beta = (W/Q)^2 $"], loc="lower right")
    plt.title("Neural network predictions")
    plt.savefig("steady_figures/steady_k_nn.png")
    plt.show()

    plt.figure(figsize=(8,5)) 
    plt.plot(X_u_train, u_train, 'ok') 
    plt.plot(L-x, u_exact, '-k', linewidth=2)
    plt.plot(L-x, u_none, '--r', linewidth=2)
    plt.plot(L-x, u_lin, '--g', linewidth=2)
    plt.plot(L-x, u_quad, '--b', linewidth=2)
    plt.xlabel("x")
    plt.ylabel("h") 
    plt.grid(True, which="both")
    plt.legend(["Data", "With true K", "$\\alpha = \\beta = 1 $", "$\\alpha = \\beta = W/Q$", "$\\alpha = \\beta = (W/Q)^2 $"], loc="lower right")
    plt.title("PDE solution with recovered values")
    plt.savefig("steady_figures/steady_k_pde.png")
    plt.show()



"""
    f_left, f_right = model.predictbc() 


    # predictions
    u_pred = model.predict(X_u_train) 

    # test data  
    if flip_domain:
        X_test, u_test = analytic.make_training_data(n_test) 
    else:
        X_test = np.random.uniform(high=L, size=n_test).reshape([n_test, 1])
        u_test = seepage(X_test, L, W, Q, hs, k)
    
    # predictions
    u_pred_test = model.predict(X_test)

    plt.rcParams.update({"font.size" : 16}) 
    plt.figure(figsize=(8,5))
    plt.plot

    # plot data vs trained predictions
    plt.figure(figsize=FIGSIZE)
    plt.subplot(211)
    plt.plot(X_u_train, u_train, 'or')
    plt.plot(X_u_train, u_pred[0], 'ob') 
    plt.xlabel("x")
    plt.ylabel("h")
    plt.ylim([0,None])
    plt.grid(True, which="both")
    plt.title("u (training data)") 
    plt.legend(["Data", "Prediction"])
    
    plt.subplot(212)
    plt.plot(X_test, u_test, 'or')
    plt.plot(X_test, u_pred_test[0], 'ob') 
    plt.xlabel("x")
    plt.ylabel("h")
    plt.ylim([0,None])
    plt.grid(True, which="both")
    plt.title("u (test data)") 
    plt.legend(["Data", "Prediction"])

    # plot some residuals 
    plt.figure(figsize=FIGSIZE)
    plt.subplot(211)
    plt.semilogy(X_u_train, np.abs(u_pred[1]), 'ob') 
    plt.xlabel("x")
    plt.ylabel("residual")
    plt.grid(True, which="both")
    plt.title("f (training data)") 
    
    plt.subplot(212)
    plt.semilogy(X_test, np.abs(u_pred_test[1]), 'ob') 
    plt.xlabel("x")
    plt.ylabel("residual")
    plt.grid(True, which="both")
    plt.title("f (test data)") 
    plt.show()
"""
