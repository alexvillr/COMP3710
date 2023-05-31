import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
print(f"TF Version: {tf.__version__}")

#base layout
Y, X = np.mgrid[-1.3:1.3:0.0005, -2:1:0.0005]
Z = X+1j*Y
#for different zoom
def zoom (Z, offset, yoffset, zoom):
    '''
    takes in how much you want to offset the fractal to the left, and then how many times bigger you want to make the fractal
    '''
    return ((Z - offset + 1j * yoffset) * 1/zoom)

# Z = zoom(Z, 8, 5.5)

xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, tf.float32))

#mandelbrot
for i in range(200):
    #Compute the new values of z: z^2 + x
    zs_ = zs*zs + xs
    # Have we diverged with this new value?
    not_diverged = tf.abs(zs_) < 4
    # Update variables to compute
    ns.assign_add(tf.cast(not_diverged, tf.float32))
    zs.assign(zs_)


fig = plt.figure(figsize=(16,10))
def processFractal(a):
    """Display an array of iteration counts as a
    colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a
    
plt.imshow(processFractal(ns.numpy()))
plt.tight_layout(pad=0)
plt.show()


