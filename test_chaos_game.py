"""
script for playing the chaos game
"""
import numpy as np
import matplotlib.pylab as plt

n = 3

#generate n points on a unit circle
r = np.arange(0, n)
points = np.exp(2.0 * np.pi * 1j * r / n)
print(points)

#plot points
res = 100
w = np.arange(0, res)
circle_points = np.exp(2.0 * np.pi * 1j * w / res)

#starting point
start = 0.1 + 0.5j

plt.plot(np.real(circle_points), np.imag(circle_points), "b-")
plt.plot(np.real(points), np.imag(points), "r.")
plt.plot(np.real(start), np.imag(start), "g.")



plt.show()