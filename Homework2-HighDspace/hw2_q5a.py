#class   : Foundations of ML and Data Science
#homework: 2
#question: 5(a)
#author  : Mohammad Afzal Shadab
#email   : mashadab@utexas.edu

#High dimensional unit ball
from numpy import random, linalg, zeros_like
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

d    = 2       #dimensions
N    = 1000    #number of random points
A    = random_ball(N, d, ) #generating the random ball

plt.figure(figsize=(15,7.5) , dpi=100)    
plt.scatter(A[:,0],A[:,1],color='red')
plt.axis('scaled')
plt.tight_layout()
plt.savefig(f'random_ball_N{N}_dimension{d}.png',bbox_inches='tight', dpi = 600)
