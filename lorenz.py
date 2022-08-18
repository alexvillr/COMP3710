import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# https://doi.org/10.1016/0375-9601(83)90052-X

# using the sigma, rho and beta values that make the lorenz attractor a fractal
# according to wikipedia
def lorenz(x, y, z, sigma=16.0, rho=40.0, beta=4.0):
    # lorenz equations
    x_dot = sigma * (y - x)
    y_dot = rho * x - y - x * z
    z_dot = x * y - beta * z

    return x_dot, y_dot, z_dot

# setting dt and number of steps for appropriate clarity
dt=0.01
num_steps = 20000

# initialise empty array and initial values for a clean spiral
xs, ys, zs = np.empty(num_steps + 1), np.empty(num_steps + 1), np.empty(num_steps + 1)
xs[0], ys[0], zs[0] = 0.0, 1.0, 1.05

# itterate over array calculating what the new value should be in each case based on previous values
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    xs[i+1] = xs[i] + x_dot * dt
    ys[i+1] = ys[i] + y_dot * dt
    zs[i+1] = zs[i] + z_dot * dt


# do a 3d projection of the plot making it pretty with some colour choices
fig = plt.subplot(projection="3d")
fig.plot(xs, ys, zs, lw=0.5, color="navy")
fig.scatter(xs, ys, zs, lw=0.1, alpha=0.1, color="red")

plt.show()