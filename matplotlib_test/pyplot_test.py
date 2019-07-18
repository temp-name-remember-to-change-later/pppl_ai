import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

n = 1000
x = lambda t: 3 * np.sin(t / 10.)
y = lambda t: 1.5 * np.sin((2.01) * t/10)
z = lambda t: 1.6 * np.sin((3) * t/10)
xin = np.array([2*x(t+0.2) for t in range(n)])
yin = np.array([y(t+1) for t in range(n)])
zin = np.array([z(t) for t in range(n)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', proj_type='persp')
Axes3D.bar3d(ax, xin, yin, zin, 0.1, 0.1, 0.1)
plt.show()



