# galaxy data
# venv activate -> $source venv/bin/activate

# N: 입자 수
# location [x, y, z]
# velocity [vx, vy, vz]
# mass     m_B 


import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

N = 1000

### galaxy A
# loc
x_A = np.random.normal(loc=10.0, scale=1.5, size=N)
y_A = np.random.normal(loc=0.0, scale=0.15, size=N)
z_A = np.random.normal(loc=0.0, scale=1.5, size=N)
# velocity
vx_A = np.full(N, -0.5)
vy_A = np.full(N)
vz_A = np.full(N)
# mass
m_A = np.ones(N)


### galaxy B
# loc
x_B = np.random.normal(loc=-10.0, scale=1.5, size=N)
y_B = np.random.normal(loc=0.0, scale=0.15, size=N)
z_B = np.random.normal(loc=0.0, scale=1.5, size=N)
# velocity
vx_B = np.full(N, 0.5)
vy_B = np.full(N)
vz_B = np.full(N)
# mass
m_A = np.ones(N)


### 2D data vizualization
#plt.scatter(x_A, z_A, s=0.5, c='blue', label='Galaxy A')
#plt.scatter(x_B, z_B, s=0.5, c='red', label='Galaxy B')
#plt.legend()
#plt.show()
###

### 3D data vizualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_A, y_A, z_A, s=0.5, c='blue', label='Galaxy A')
ax.scatter(x_B, y_B, z_B, s=0.5, c='red', label='Galaxy B')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
###
