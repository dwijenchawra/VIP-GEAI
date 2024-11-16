import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate two random quaternions
q1 = np.random.rand(4)
q2 = np.random.rand(4)

# Normalize the quaternions to ensure they are unit quaternions
q1 = q1 / np.linalg.norm(q1)
q2 = q2 / np.linalg.norm(q2)

# SLERP between two quaternions using Slerp class from SciPy
slerp = Slerp(rotation1=Rotation.from_quat(q1), rotation2=Rotation.from_quat(q2))

# Create a figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the two quaternions as vectors in 3D space
q1_line, = ax.plot([0, Rotation.from_quat(q1).apply([0, 0, 0], [0, 1, 0])], 'b-')
q2_line, = ax.plot([0, Rotation.from_quat(q2).apply([0, 0, 0], [0, 1, 0])], 'r-')

# Animate the SLERP
for i in range(100):
    t = i / 100.0
    interpolated_rotation = slerp(t)
    
    ax.clear()
    q_slerp_line, = ax.plot([0, interpolated_rotation.apply([0, 0, 0], [0, 1, 0])], 'b-')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.pause(0.05)

# Set the axis limits and labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()