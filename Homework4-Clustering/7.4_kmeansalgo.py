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
p = 0.7; q = 0.3

'''
Asub = p*np.ones((50,50))
I    = np.eye(3)
A    = np.kron(I,Asub)
A[A==0] = q 
'''
A = 0.3*np.ones((150,150), dtype = float)
A[:50,:50] = 0.7
A[50:100, 50:100] = 0.7
A[100:,100:] = 0.7

B = np.random.uniform(0,1,(150,150))

'''
for i in range(np.shape(A)[0]):     #columns
    for j in range(np.shape(A)[1]): #rows
        if (B[i,j]>=A[i,j]):
            A[i,j] = 1.0
        else:
            A[i,j] = 0.0
'''

plt.figure()
plt.imshow(A)
plt.title('Initial A')
print('Initial nonzero',np.count_nonzero(A))
A = B > A

plt.figure()
plt.imshow(A)
plt.title('Filtered A')
print('Filtered nonzero',np.count_nonzero(A))

def permute_rows_cols(A):
    A = np.random.permutation(A) #randomly permutating the elements of A in rows
    A = np.transpose(A) #transposing to do the permutation in first index
    A = np.random.permutation(A) #randomly permutating the elements of A in columns
    A = np.transpose(A)  #transposing to do get the final permuted matrix
    return A

A = permute_rows_cols(A)

plt.figure()
plt.imshow(A)
plt.title('Permuted A')

#x = np.argwhere(A==1)[:,0]; y=np.argwhere(A==1)[:,1] 
df = pd.DataFrame({
        'x':np.argwhere(A==1)[:,0],
        'y':np.argwhere(A==1)[:,1] 
})
np.random.seed(1000)

k = 10
#centroids[i] = [x,y]
centroids = {
        i+1: [np.random.randint(0,150),np.random.randint(0,150)]
        for i in range(k)
} 

fig = plt.figure(figsize=(10,10))
plt.scatter(df['x'],df['y'],color='k')
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'c', 5: 'm', 6: 'y', 7: 'darkblue', 8: 'orange', 9: 'deeppink', 10: 'maroon'}
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,150)
plt.ylim(0,150)
plt.show()

#Assignment stage
def assignment(df,centroids):
    for i in centroids.keys():
        #sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                    (df['x'] - centroids[i][0]) **2.0
                    + (df['y'] - centroids[i][1]) **2.0
                    )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color']   = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
print(df.head())

fig = plt.figure(figsize=(10,10))
plt.title('Initialization')
plt.scatter(df['x'],df['y'], color = df['color'], alpha=0.3, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i],s=200)
plt.xlim(0,150)
plt.ylim(0,150)
plt.show()

#Update Stage
old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)

fig = plt.figure(figsize=(10,10))
plt.title('Update')
ax  = plt.axes()
plt.scatter(df['x'],df['y'], color = df['color'], alpha=0.3, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i],s=200)
plt.xlim(0,150)
plt.ylim(0,150)

for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0])*0.75
    dy = (centroids[i][1] - old_centroids[i][1])*0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc = colmap[i], ec= colmap[i])
plt.show()

#Repeat until convergence
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df,centroids)
    if closest_centroids.equals(df['closest']):
        break
fig = plt.figure(figsize=(10,10))
plt.title('Final Result')
plt.scatter(df['x'],df['y'], color = df['color'], alpha=0.3, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i],s=200)
plt.xlim(0,150)
plt.ylim(0,150)
plt.show()   


        