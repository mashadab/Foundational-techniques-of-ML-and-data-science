#class   : Foundations of ML and Data Science
#homework: 4
#question: 4
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Serif"


#k-means algorithm
def kmeans(data, k, random_permutation, max_iters = 10000):

    ''' 
    INPUT
    data: nxd data matrix 
    k: number of clusters
    random_permutation: The indices of how data shuffling. Will be used to unshuffle the labels.
    max_iters: maximum iterations of the algorithm
    OUTPUT
    unshuffled_labels:
    sum_of_squares: Sum of the squares to the cluster centers
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
            print(f"Lloyd's k-means algorithm has converged in {iters+1} steps.")
            break

    sum_of_squares = 0.
    unshuffled_labels = np.zeros((n), dtype = int)
    for i in range(n):
        unshuffled_labels[random_permutation[i]] = labels[i]
        sum_of_squares = sum_of_squares + np.linalg.norm(A_permuted[i,:]-centres[int(labels[i])])**2

    return unshuffled_labels, sum_of_squares

#Initializing A
p = 0.7; q = 0.3
Asub = p*np.ones((50,50))
I    = np.eye(3)
A    = np.kron(I,Asub)
A[A==0] = q 

B = np.random.uniform(0,1,(150,150)) 
#filtering matrix

plt.figure()
plt.imshow(A)
plt.title('Initial A')
A = B > A  #filtering A through B

plt.figure()
plt.imshow(A)
plt.title('Filtered A')

random_permutation = np.random.permutation(150) #generating a random permutation

plt.figure()
A_permuted = A[:,random_permutation]
A_permuted = A_permuted[random_permutation,:]
plt.imshow(A_permuted)
plt.title("Permuted and filtered Matrix A")

#Part a
plt.figure()
unshuffled_labels_k3, _ = kmeans(A_permuted, 3, random_permutation)
plt.figure()
plt.scatter(np.arange(150), unshuffled_labels_k3, color='r')
plt.xlabel("Labels for unscrambled rows of A for k = 3")
plt.ylabel("Cluster number")
plt.savefig('Q4parta.pdf')

#Part b
k_array = np.arange(1,11)
sum_of_squares_array = np.zeros((10))
print('Cluster number, Elements in cluster, Square distance:')
for i in range(10):
    unshuffled_labels, sum_of_squares_array[i] = kmeans(A_permuted, i+1, random_permutation)
    print(np.unique(unshuffled_labels, return_counts=True)[0]+1,np.unique(unshuffled_labels, return_counts=True)[1], sum_of_squares_array[i])
    
plt.figure()
plt.plot(k_array, sum_of_squares_array,'r')
plt.xlabel('k')
plt.ylabel('Sum of squares from centres')
plt.savefig('Q4partb.pdf')