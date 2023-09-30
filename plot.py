#%%
from sys import stdin
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd

fig, ax = plt.subplots()
plt.show(block=False)

xs = []
ys = []

#%%
def animate(i):
    row = next(stdin).rstrip().split()
    temp, step, energy, magnet = map(float, row)

    xs.append(step)
    ys.append(magnet)

    ax.scatter(xs, ys)


anim = FuncAnimation(plt.gcf(), animate, interval=1)

plt.show()
