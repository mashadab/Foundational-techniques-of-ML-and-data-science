#class   : Foundations of ML and Data Science
#homework: 4
#question: 2
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Serif"

def create_sheath(n=100):
    
    #create n data points on a one dimensional sheath in three dimensional space
    data = np.zeros((3,n))
    for i in range(n):
        x = np.sin(np.pi/100*(i+1))
        y = np.sqrt(1-x**2.0)    
        z = 0.003*(i+1)  
        data[:,i] = [x,y,z]
    
    #subtract adjacent vertices
    distance = np.zeros((9,3,10))
    for j in range(9):
        for i in range(j*10,j*10+5):
            distance[j,:,i-j*10] = data[:,i]-data[:,j*10+5]
            distance[j,:,i+5-j*10] = data[:,i+6]-data[:,j*10+5] 
    return data, distance

data, distance = create_sheath(100)

plt.figure(1)
for j in range(9):
    u, s, vh = np.linalg.svd(distance[j,:,:], full_matrices=False)
    plt.plot(s,label=f'i={j*10}-{j*10+10}')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Singular value')
plt.savefig('Q2.pdf')