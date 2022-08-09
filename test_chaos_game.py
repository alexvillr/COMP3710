"""
script for playing the chaos game
"""
from xmlrpc.client import MAXINT
import numpy as np
import matplotlib.pylab as plt

n = 4

#generate n points on a unit circle
r = np.arange(0, n)
points = np.exp(2.0 * np.pi * 1j * r / n)
print(points)

#plot points
res = 1000
w = np.arange(0, res)
circle_points = np.exp(2.0 * np.pi * 1j * w / res)

#starting point
start = 0.1 + 0.5j

# plt.plot(np.real(circle_points), np.imag(circle_points), "b-")
# plt.plot(np.real(points), np.imag(points), "r.")

#play the game
select = np.random.randint(0, n)
print(points[select])

#new point
new_point = points[select] - start
new_point += start

#plot it

#full algorithm
def compute(startloc, last_point=points[-1]):
    '''
    compute new position for game
    '''
    randloc = np.random.randint(0, n)
    while (points[randloc] == last_point):
        randloc = np.random.randint(0, n)
    new_point = points[randloc] - startloc
    new_point = startloc + new_point / 2
    return new_point, points[randloc]

pl, rloc = compute(start)

# plt.plot(np.real(circle_points), np.imag(circle_points), "b-")
# plt.plot(np.real(points), np.imag(points), "r.")
# plt.plot(np.real(start), np.imag(start), "g.")
iterations = 100000

next_point = start
for i in range(iterations):
    next_point, rloc = compute(next_point, rloc)
    plt.plot(np.real(next_point), np.imag(next_point), "b.")


plt.show()