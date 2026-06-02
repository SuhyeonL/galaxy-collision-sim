# vizualization

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# CPU 결과 시각화
with open("../cpu/viz_data.bin", "rb") as f:
    STEPS = np.frombuffer(f.read(4), dtype=np.int32)[0]
    N = np.frombuffer(f.read(4), dtype=np.int32)[0]
    position = np.frombuffer(f.read(), dtype=np.float32)


# print(STEPS)
# print(N)
# print(len(position))
# 3D 배열로 변환
position = position.reshape(STEPS, N, 3)

# 위치 변화 시각화
fig, ax = plt.subplots()

def update(step):
    # 이전 스텝의 점들을 지움
    ax.clear()
    # 은하 A의 N개 입자들의 x, y, z를 한번에 update
    x_A = position[step][:N//2, 0]
    y_A = position[step][:N//2, 1]
    z_A = position[step][:N//2, 2]

    # 은하 B의 N개 입자들의 x, y, z를 한번에 update
    x_B = position[step][N//2:, 0]
    y_B = position[step][N//2:, 1]
    z_B = position[step][N//2:, 2]

    # 2D 시각화 
    # x, z 좌표만 사용
    ax.scatter(x_A, z_A, s=0.5, c='blue', label='Galaxy A')
    ax.scatter(x_B, z_B, s=0.5, c='orange', label='Galaxy B')


    

animation = FuncAnimation(fig, update, frames=STEPS, interval=50)
plt.show()

