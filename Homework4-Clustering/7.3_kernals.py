#class   : Foundations of ML and Data Science
#homework: 4
#question: 3
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#importing libraries
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
plt.rcParams["font.family"] = "Serif"

x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)

X,Y = np.meshgrid(x,y)

Xcol = np.reshape(X,-1)
Ycol = np.reshape(Y,-1)

Xcircle = Xcol[Xcol**2.0 + Ycol**2.0<=1.0]
Ycircle = Ycol[Xcol**2.0 + Ycol**2.0<=1.0]

#Xouter = [i for i in Xcol if i not in Xcircle]
#Youter = [i for i in Ycol if i not in Ycircle]

K = (np.transpose(X) @ X + np.transpose(Y) @ Y + 1 )**2.0

X_points = X[K<1000]
Y_points = Y[K<1000]

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, K,cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('K')
ax.set_title('Surface plot')
plt.show()

fig = plt.figure()
plt.scatter(X_points,Y_points,color='k')
plt.contour(X, Y, K, 20, cmap='RdGy')
clb = plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
clb.set_label('K')
plt.savefig('Q3b.pdf')