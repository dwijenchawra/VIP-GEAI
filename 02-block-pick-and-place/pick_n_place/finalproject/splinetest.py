import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


# 3D example
total_rad = 10
z_factor = 3
noise = 0.1

num_true_pts = 200
s_true = np.linspace(0, total_rad, num_true_pts)
x_true = np.cos(s_true)
y_true = np.sin(s_true)
z_true = s_true/z_factor

# num_sample_pts = 80
# s_sample = np.linspace(0, total_rad, num_sample_pts)
# x_sample = np.cos(s_sample) + noise * np.random.randn(num_sample_pts)
# y_sample = np.sin(s_sample) + noise * np.random.randn(num_sample_pts)
# z_sample = s_sample/z_factor + noise * np.random.randn(num_sample_pts)
points = [
    [-0.5  , 0.65 , 1.75],
    [0  , 0.65 , 1.75],
    [0.5  , 0.65 , 1.75],
    [0.5  , 0.65 , 1.5],
    [0.  , 0.65 , 1.5],
    [-0.5  , 0.65 , 1.5],
    [-0.5  , 0.65 , 1.25],
    [0.  , 0.65 , 1.25],
    [0.5 , 0.65 , 1.25]
]


x_sample = [p[0] for p in points]
y_sample = [p[1] for p in points]
z_sample = [p[2] for p in points]

tck, u = interpolate.splprep([x_sample,y_sample,z_sample], s=3)
x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
u_fine = np.linspace(0,1,num_true_pts)
x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)

fig2 = plt.figure(2)
ax3d = fig2.add_subplot(111, projection='3d')
# ax3d.plot(x_true, y_true, z_true, 'b')
ax3d.plot(x_sample, y_sample, z_sample, 'r*')
ax3d.plot(x_knots, y_knots, z_knots, 'go')
ax3d.plot(x_fine, y_fine, z_fine, 'go')
fig2.show()
plt.show()
