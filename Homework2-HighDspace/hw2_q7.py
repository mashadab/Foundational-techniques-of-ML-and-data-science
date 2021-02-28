#class   : Foundations of ML and Data Science
#homework: 2
#question: 7(a),(b)
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#High dimensional unit ball
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


# To get a random matrix of num_vectors in dimension
# Input : dimensions and number of vectors
# Output: num_vectors X dimension unit Gaussian random matrix
def random_Gaussian_vectors(dimension, num_vectors):
    return np.random.normal(0,1,(dimension,num_vectors))

# This function stores the projection of vector mat2 from R^{dimensions} 
# to R^{reduced_num_vectors}
# Input :mat1 (dim X reduced_num_vectors) helps with the projection
#        mat2 (dim X num_vectors) is being projected onto R^{reduced_num_vectors}
# Output: projection_matrix (reduced_num_vectors X num_vectors) 
def projection_matrix(mat1,mat2):
    dimension, num_vectors = np.shape(mat1)
    reduced_num_vectors    = np.shape(mat2)[1]
    
    projection_matrix = np.empty((num_vectors,reduced_num_vectors))
    
    for i in range(0,num_vectors):
        for j in range(0,reduced_num_vectors):
            projection_matrix[i,j] = np.dot(mat1[:,i], mat2[:,j]) #projection of columns
    return projection_matrix
    
# This function computes the dot product all possible pair of n vectors
# Input : mat (reduced_num_vectors X num_vectors)
# Output: a matrix of angles with columns (i,j,angles)
def comp_angle_matrix(mat):
    angle_matrix  = []
    num_vectors   = np.shape(mat)[1]
    
    for i in range(0,num_vectors):
        for j in range(0,num_vectors):
            angle = np.arccos(np.dot(mat[:,i], mat[:,j])/(np.linalg.norm(mat[:,i])*np.linalg.norm(mat[:,j]))) #calculating the angles
            if(i<j):
                angle_matrix.append([i+1,j+1,angle])
    return np.array(angle_matrix)

#Part (a)
parta_start = perf_counter()
A  =  random_Gaussian_vectors(1000,100)  #random 1000-dimensional space
B  =  random_Gaussian_vectors(1000,1000)
B,dummy = np.linalg.qr(B)                #finding 1000 orthonormal 1000-dimensional vectors using QR

projMat = projection_matrix(A,B)
angleMat= comp_angle_matrix(projMat)

mean = np.mean(angleMat[:,2]); std = np.std(angleMat[:,2])
parta_stop = perf_counter()

print('The true value is \u03C0/2 = ', np.pi/2)
print(f'Part (a): The value with random projection is = {mean} \u00B1 {std}')

#Part (b)
partb_start = perf_counter()
A  =  random_Gaussian_vectors(100,1000)  #random 1000-dimensional space
angleMat= comp_angle_matrix(A)
mean = np.mean(angleMat[:,2]); std = np.std(angleMat[:,2])
partb_stop = perf_counter()
print(f'Part (b): The value with Gaussian vectors is = {mean} \u00B1 {std}')
print('\n Run times \n')
print(f'Running part (a) took {parta_stop - parta_start} seconds \n')
print(f'Running part (b) took {partb_stop - partb_start} seconds')