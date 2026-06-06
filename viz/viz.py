# viz.py
# 은하 충돌 시뮬레이션 시각화
#
# 실행 방법: viz/ 디렉터리에서
#   python viz.py
#
# 구성:
#   - 왼쪽: 3D scatter 애니메이션 (은하 A/B 궤적)
#   - 오른쪽: XZ 평면 입자 밀도 히트맵 (충돌 에너지 분포 분석)

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

# ── 데이터 로드 ───────────────────────────────────
# GPU 결과 우선, 없으면 CPU 결과 사용
for path in ["../gpu/viz_data.bin", "../cpu/viz_data.bin"]:
    if os.path.exists(path):
        DATA_PATH = path
        print(f"데이터 파일: {DATA_PATH}")
        break
else:
    raise FileNotFoundError("viz_data.bin 없음. CPU 또는 GPU 시뮬레이션을 먼저 실행하세요.")

with open(DATA_PATH, "rb") as f:
    STEPS = np.frombuffer(f.read(4), dtype=np.int32)[0]
    N     = np.frombuffer(f.read(4), dtype=np.int32)[0]
    raw   = np.frombuffer(f.read(), dtype=np.float32)

# shape: (STEPS, N, 3)
position = raw.reshape(STEPS, N, 3)
half = N // 2   # Galaxy A: [0, half), Galaxy B: [half, N)

print(f"STEPS={STEPS}, N={N}")

# ── 히트맵 설정 ───────────────────────────────────
HEATMAP_BINS = 80      # 해상도
AXIS_RANGE   = 20.0    # 축 범위

def compute_density(step: int) -> np.ndarray:
    """XZ 평면 기준 2D 밀도 히스토그램 (전체 입자)"""
    x = position[step][:, 0]
    z = position[step][:, 2]
    H, _, _ = np.histogram2d(
        x, z,
        bins=HEATMAP_BINS,
        range=[[-AXIS_RANGE, AXIS_RANGE], [-AXIS_RANGE, AXIS_RANGE]]
    )
    return H.T   # imshow는 (row=y, col=x) 순서

# colormap 스케일 고정: 전체 스텝의 최대 밀도 샘플링
sample_steps = range(0, STEPS, max(1, STEPS // 30))
max_density  = max(compute_density(s).max() for s in sample_steps) + 1

# ── Figure 구성 ───────────────────────────────────
fig = plt.figure(figsize=(14, 6), facecolor="black")
fig.suptitle("Galaxy Collision Simulation", color="white", fontsize=13, y=0.98)

# 왼쪽: 3D scatter
ax3d = fig.add_subplot(121, projection="3d")
ax3d.set_facecolor("black")

# 오른쪽: 히트맵
ax2d = fig.add_subplot(122)
ax2d.set_facecolor("black")
ax2d.set_xlabel("X", color="gray")
ax2d.set_ylabel("Z (depth)", color="gray")
ax2d.set_title("입자 밀도 히트맵 (XZ 평면)", color="white", fontsize=11)
ax2d.tick_params(colors="gray")

# 히트맵 초기 렌더링
H0 = compute_density(0)
im = ax2d.imshow(
    H0,
    origin="lower",
    extent=[-AXIS_RANGE, AXIS_RANGE, -AXIS_RANGE, AXIS_RANGE],
    cmap="inferno",
    norm=mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=max_density),
    interpolation="gaussian",
    aspect="equal"
)
cbar = fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color="gray")
cbar.set_label("입자 수 (밀도)", color="gray", fontsize=9)

step_label_2d = ax2d.text(0.02, 0.96, "", transform=ax2d.transAxes,
                           color="white", fontsize=9, verticalalignment="top")

# ── 애니메이션 업데이트 함수 ─────────────────────
def update(step):
    # ── 3D scatter 업데이트 ──
    ax3d.clear()
    ax3d.set_facecolor("black")
    ax3d.set_xlim(-AXIS_RANGE, AXIS_RANGE)
    ax3d.set_ylim(-AXIS_RANGE, AXIS_RANGE)
    ax3d.set_zlim(-AXIS_RANGE, AXIS_RANGE)
    ax3d.set_xlabel("X", color="gray", labelpad=2)
    ax3d.set_ylabel("Y", color="gray", labelpad=2)
    ax3d.set_zlabel("Z", color="gray", labelpad=2)
    ax3d.tick_params(colors="gray", labelsize=6)

    x_A = position[step][:half, 0]
    y_A = position[step][:half, 1]
    z_A = position[step][:half, 2]
    x_B = position[step][half:, 0]
    y_B = position[step][half:, 1]
    z_B = position[step][half:, 2]

    ax3d.scatter(x_A, y_A, z_A, s=0.3, c="#4fc3f7", alpha=0.7, label="Galaxy A")
    ax3d.scatter(x_B, y_B, z_B, s=0.3, c="#ff8a65", alpha=0.7, label="Galaxy B")
    ax3d.legend(loc="upper right", fontsize=7, framealpha=0.3,
                labelcolor="white", facecolor="black")
    ax3d.text2D(0.02, 0.95, f"Step {step} / {STEPS - 1}",
                transform=ax3d.transAxes, color="white", fontsize=9)

    # ── 히트맵 업데이트 ──
    H = compute_density(step)
    im.set_data(H)
    step_label_2d.set_text(f"Step {step} / {STEPS - 1}")

    return [im]


# ── 실행 ─────────────────────────────────────────
ani = FuncAnimation(
    fig, update,
    frames=STEPS,
    interval=40,
    blit=False
)

plt.tight_layout()
plt.show()
