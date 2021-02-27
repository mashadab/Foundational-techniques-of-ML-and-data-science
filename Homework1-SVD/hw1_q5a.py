#class   : Foundations of Data Science
#homework: 1
#question: 5
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
plt.rcParams['figure.figsize'] = [16.0, 8.0]
plt.rcParams['figure.dpi'] = 300

#Part (a) Image analysis using SVD
A = imread('austin_image.jpg')  #reading the image
X = np.mean(A, -1)              #converting RGB to grayscale

img = plt.imshow(X)
plt.set_cmap('gray')
plt.axis('off')
plt.savefig(f'original.jpg')

U, S, VT = np.linalg.svd(X, full_matrices =False) #Performing reduced SVD
S        = np.diag(S)                             #converting into diagonal matrix 

j = 0

for r in (1,4,16,32):
    #Constructing a low rank approximation of the image
    Xapprox = U[:,:r] @ S[:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    img = plt.imshow(Xapprox)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.title(f' r= {r}')
    plt.savefig(f'svd_{r}modes.jpg')
    j += 1

plt.figure(j+1)
plt.semilogy(np.diag(S))
plt.ylabel('Singular values')
plt.xlabel('Number of modes')
plt.savefig('singular_values.jpg')

plt.figure(j+2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.ylabel('Cumulative sum of Singular values')
plt.xlabel('Number of modes')
plt.savefig('cum_sum_singular_values.jpg')


#Part (b) Percentage of Frobenius norm
S = np.diag(S)

norm = np.zeros((4,2))
j = 0
for r in (1,4,16,32):
    norm[j,0] = int(r)
    norm[j,1] = np.linalg.norm(S[:r])/np.linalg.norm(S)*100
    j += 1

print('Modes         Frobenius norm \n')
print(norm)


#Part (c) White noise image analysis using SVD
X   = np.random.rand(*X.shape)

img = plt.imshow(X)
plt.set_cmap('gray')
plt.axis('off')
plt.savefig(f'noise_original.jpg')

U, S, VT = np.linalg.svd(X, full_matrices =False) #Performing reduced SVD
S        = np.diag(S)                             #converting into diagonal matrix 

j = 0

for r in (1,4,16,32):
    #Constructing a low rank approximation of the image
    Xapprox = U[:,:r] @ S[:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    img = plt.imshow(Xapprox)
    plt.set_cmap('gray')
    plt.axis('off')
    plt.title(f' r= {r}')
    plt.savefig(f'noise_svd_{r}modes.jpg')
    j += 1

plt.figure(j+1)
plt.semilogy(np.diag(S))
plt.ylabel('Singular values')
plt.xlabel('Number of modes')
plt.savefig('noise_singular_values.jpg')

plt.figure(j+2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.ylabel('Cumulative sum of Singular values')
plt.xlabel('Number of modes')
plt.savefig('noise_cum_sum_singular_values.jpg')

S = np.diag(S)

norm = np.zeros((4,2))
j = 0
for r in (1,4,16,32):
    norm[j,0] = int(r)
    norm[j,1] = np.linalg.norm(S[:r])/np.linalg.norm(S)*100
    j += 1
print('White noise \n')
print('Modes         Frobenius norm \n')
print(norm)