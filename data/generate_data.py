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
vy_A = np.zeros(N)
vz_A = np.zeros(N)
# mass
m_A = np.ones(N)


### galaxy B
# loc
x_B = np.random.normal(loc=-10.0, scale=1.5, size=N)
y_B = np.random.normal(loc=0.0, scale=0.15, size=N)
z_B = np.random.normal(loc=0.0, scale=1.5, size=N)
# velocity
vx_B = np.full(N, 0.5)
vy_B = np.zeros(N)
vz_B = np.zeros(N)
# mass
m_B = np.ones(N)


### 2D data vizualization
#plt.scatter(x_A, z_A, s=0.5, c='blue', label='Galaxy A')
#plt.scatter(x_B, z_B, s=0.5, c='red', label='Galaxy B')
#plt.legend()
#plt.show()
### 

### 3D data vizualization
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x_A, y_A, z_A, s=0.5, c='blue', label='Galaxy A')
# ax.scatter(x_B, y_B, z_B, s=0.5, c='red', label='Galaxy B')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.legend()
# plt.show()
###

### 데이터 구조화
gal_A = np.column_stack((x_A, y_A, z_A, vx_A, vy_A, vz_A, m_A))
gal_B = np.column_stack((x_B, y_B, z_B, vx_B, vy_B, vz_B, m_B))

# [0 ~ 999]     -> A
# [1000 ~ 1999] -> B
total_gal = np.vstack((gal_A, gal_B))

# .bin 파일 생성
data = np.concatenate((np.array([N]), total_gal.flatten()))
data.tofile("gal_data.bin")