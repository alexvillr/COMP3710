import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# https://doi.org/10.1016/0375-9601(83)90052-X

def lorenz(x, y, z, sigma=16.0, rho=40.0, beta=4.0):
    x_dot = sigma * (y - x)
    y_dot = rho * x - y - x * z
    z_dot = x * y - beta * z

    return x_dot, y_dot, z_dot

dt = 0.01
num_steps = 40
X, Y, Z = np.arange(0.0, num_steps + 1.0, 0.01), np.arange(0.0, num_steps + 1.0, 0.01), np.arange(0.0, num_steps + 1.0, 0.01)

xs = tf.constant(X.astype(np.float32))
ys = tf.constant(Y.astype(np.float32))
zs = tf.constant(Z.astype(np.float32))

for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    Is = xs + (x_dot * dt)
    Js = ys + (y_dot * dt)
    Ks = zs + (z_dot * dt)

xs = Is.numpy()
ys = Js.numpy()
zs = Ks.numpy()

fig = plt.subplot(projection="3d")
fig.plot(xs, ys, zs, lw=0.5, color="navy")
fig.scatter(xs, ys, zs, lw=0.1, alpha=0.1, color="red")

plt.show()























# X = np.arange(1.0, 40.0, 0.01)
# xs = tf.constant(X.astype(np.int32))
# xs = tf.Variable([0.0, 1.0, 1.05])
# ns = tf.Variable(tf.zeros_like(xs, tf.float32))

# def lorenz(state, sigma=16.0, rho=40.0, beta=4.0, dt=0.01):
#     x, y, z = state[0], state[1], state[2]#deconstruct current state

#     #lorenz equation
#     x_dot = sigma*(y - x)
#     y_dot = rho*x - y - x*z
#     z_dot = x*y - beta*z

#     #use derivatives to find next state
#     newx = (x + x_dot) * dt
#     newy = (y + y_dot) * dt
#     newz = (z + z_dot) * dt

#     return [newx, newy, newz]

# for i in range(40):
#     #compute next state
#     xs_ = lorenz(xs)

#     xs.assign(xs_)
    

# plt.imshow(ns.numpy())
# plt.tight_layout()
# plt.show()


# fig = plt.subplot(projection="3d")
# fig.plot(states[:, 0], states[:, 1], states[:, 2], lw=0.5, color="navy")
# fig.scatter(states[:, 0], states[:, 1], states[:, 2], lw=0.1, alpha=0.1, color="red")

# plt.show()

# dt = 0.01

# K = np.arange(0.0, 400.0, 0.01)
# ks = tf.constant(K.astype(np.float32))

# xs = tf.Variable(ks)
# ys = tf.Variable(ks)
# zs = tf.Variable(ks)

# for i in range(200):

#     x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
#     xs_ = xs + (x_dot * dt)
#     ys_ = ys + (y_dot * dt)
#     zs_ = zs + (z_dot * dt)

#     xs.assign(xs_)
#     ys.assign(ys_)
#     zs.assign(zs_)

# xs = xs.numpy()
# ys = ys.numpy()
# zs = zs.numpy()

# fig = plt.subplot(projection="3d")
# fig.plot(xs, ys, zs, lw=0.5, color="navy")
# fig.scatter(xs, ys, zs, lw=0.1, alpha=0.1, color="red")

# plt.show()
