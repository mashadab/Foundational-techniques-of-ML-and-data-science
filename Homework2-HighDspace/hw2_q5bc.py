#class   : Foundations of ML and Data Science
#homework: 2
#question: 5(b),(c)
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#High dimensional unit ball
from numpy import random, linalg, zeros_like, array, append, shape
import matplotlib.pyplot as plt

# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def random_ball(num_points, dimension, radius=1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = random.normal(size=(dimension,num_points))
    random_directions /= linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = random.random(num_points) ** (1/dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T

# Draw random points from a mixture distribution: with
# probability 1/2, draw at random uniformly from A, with probability
# 1/2, draw at random uniformly from B.
def draw_random_points(matA, matB, num_draw_points, probA, probB):
    mixture_dist = zeros_like(matA)
    for i in range(0,num_draw_points):
        p = random.uniform(0, 1,num_draw_points)              #generate random uniform probability 
        #getting mixture distribution values
        mixture_dist[p<=probA,:]       = matA[p<=probA,:]     
        mixture_dist[p>(1.0-probB),:]  = matB[p>(1.0-probB),:]  
        
    
    #find how many drawn points reside in A^B
    mixture_dist_intersect = []
    for i in range(0, num_draw_points):
        dist_from_A_center      = linalg.norm(mixture_dist[i,:])
        dist_from_B_center      = (mixture_dist[i,:]).copy()
        dist_from_B_center[0]   = dist_from_B_center[0]-t
        dist_from_B_center      = linalg.norm(dist_from_B_center)
        
        if dist_from_A_center <= 1.0 and dist_from_B_center <= 1.0:
                mixture_dist_intersect.append(mixture_dist[i,:])
    mixture_dist_intersect = array(mixture_dist_intersect)
    return mixture_dist,mixture_dist_intersect

d     = 2       #dimensions
N     = 10000   #number of random points
A     = random_ball(N, d, ) #generating the random ball
t     = 0.5                 #offset in the direction of first coordinate from center
B     = random_ball(N, d, ) #generating the random ball at the origin
B[:,0]= B[:,0] + t          #offsetting it in the first coordinate direction by t   
N_draw= 10000
probA = 0.5
probB = 0.5

mixture_dist, mixture_dist_intersect = draw_random_points(A, B, N_draw, probA, probB)
plt.figure(figsize=(15,7.5) , dpi=100)
plt.scatter(A[:,0],A[:,1],color='red',label = 'A')    
plt.scatter(B[:,0],B[:,1],color='blue',label = 'B')
plt.scatter(mixture_dist[:,0],mixture_dist[:,1],color='black',label = 'Mixture distribution')
plt.scatter(mixture_dist_intersect[:,0],mixture_dist_intersect[:,1],color='orange',label = 'Mixture distribution A^B')
legend = plt.legend(loc='best', shadow=False, fontsize='medium')
plt.axis('scaled')
plt.tight_layout()
plt.savefig(f'random_ball_intersect_N{N}_dimension{d}.png',bbox_inches='tight', dpi = 600)

# Part (c)
d     = [2,3,10,100]       #dimensions
num_intersect_points = zeros_like(d)
for i in range(0,len(d)):
    N     = 10000   #number of random points
    A     = random_ball(N, d[i], ) #generating the random ball
    t     = 0.5                 #offset in the direction of first coordinate from center
    B     = random_ball(N, d[i], ) #generating the random ball at the origin
    B[:,0]= B[:,0] + t          #offsetting it in the first coordinate direction by t   
    N_draw= 10000
    probA = 0.5
    probB = 0.5
    
    mixture_dist, mixture_dist_intersect = draw_random_points(A, B, N_draw, probA, probB)
    
    num_intersect_points[i], dummy = shape(mixture_dist_intersect)
    print('d:',d[i],', number of intersection drawn points: ',num_intersect_points[i])
