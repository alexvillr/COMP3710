'''
Script for generating cantor set
'''
import numpy as np
import matplotlib.pylab as plt

levels = 5
unit_interval = [0,1]
num_points = 4

def divide(interval, level=0):
    '''
    Divide interval into three parts
    '''
    plt.plot(interval, [level, level])
    segments = np.linspace(interval[0], interval[1], num_points)

divide(unit_interval)
plt.show()