#class   : Foundations of ML and Data Science
#homework: 4
#question: 7
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

#k-means algorithm
#According to question
p = 0.7; q = 0.3
Asub = p*np.ones((50,50))
I    = np.eye(3)
A    = np.kron(I,Asub)
A[A==0] = q 

B = np.random.uniform(0,1,(150,150))

plt.figure()
plt.imshow(A)
plt.title('Initial A')
A = B > A 

plt.figure()
plt.imshow(A)
plt.title('Filtered A')

random_permutation = np.random.permutation(150)

plt.figure()
A_permuted = A[:,random_permutation]
A_permuted = A_permuted[random_permutation,:]
plt.imshow(A_permuted)
plt.title("Permuted and filtered Matrix A")

def kmeans(data, k, random_permutation, max_iters = 10000):

    ''' 
    data: nxd matrix 
    k: number of clusters
    random_permutation: The indices of how data was shuffled. We will unshuffle the labels using this.
    max_iters: maximum iterations of the algorithm
    '''
    n = data.shape[0]
    d = data.shape[1]
    data = data.astype(float)
    centres = np.random.uniform(size = (k, d))
    labels = np.zeros((n), dtype = float)

    for iters in range(max_iters):

        labels_old = np.copy(labels)
        for i in range(n):
            distances = np.linalg.norm(centres - data[i,:], axis = 1)
            labels[i] = np.argmin(distances)

        for i in range(k):
            centres[i,:] = np.mean(np.squeeze(data[np.argwhere(labels==i),:]), dtype=float, axis = 0)
        
        if (np.all(labels_old == labels)):
            print("K means converged.")
            break

    sum_of_squares = 0.
    unshuffled_labels = np.zeros((n), dtype = int)
    for i in range(n):
        unshuffled_labels[random_permutation[i]] = labels[i]
        sum_of_squares = sum_of_squares + np.linalg.norm(A_permuted[i,:]-centres[int(labels[i])])**2

    return unshuffled_labels, sum_of_squares

unshuffled_labels_3, _ = kmeans_n(A_permuted, 3, random_permutation)
plt.figure()
plt.scatter(np.arange(150), unshuffled_labels_3, color='r')
plt.title("Labels for unscrambled rows of A for k = 3")


k_array = np.arange(1,11)
sum_of_squares_array = np.zeros((10))
for i in range(10):
    unshuffled_labels, sum_of_squares_array[i] = kmeans_n(A_permuted, i+1, random_permutation)
    print(np.unique(unshuffled_labels, return_counts=True), sum_of_squares_array[i])