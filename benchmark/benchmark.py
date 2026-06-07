# benchmark.py
# CPU vs GPU 성능 비교 벤치마크
#
# 실행 방법: galaxy-collision-sim-main/ 에서
#   python benchmark/benchmark.py
#
# 전제:
#   - cpu/nbody_cpu  : 빌드 완료
#   - gpu/nbody-gpu  : 빌드 완료
#   - data/generate_data.py 와 같은 포맷으로 데이터 생성

import subprocess
import os
import re
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 서버(헤드리스) 환경 — 화면 없이 PNG 저장
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 벤치마크 설정 ─────────────────────────────────
# N_LIST: 은하당 입자 수 (실제 총 입자는 2배)
# 너무 크면 CPU가 매우 오래 걸리므로 적절히 조정
N_LIST = [256, 512, 1024, 2048, 4096, 8192]

CPU_BIN = "./cpu/nbody_cpu"
GPU_BIN = "./gpu/nbody-gpu"
DATA_DIR = "./data"
BIN_FILE = os.path.join(DATA_DIR, "gal_data.bin")


# ── 데이터 생성 함수 ──────────────────────────────
def generate_data(n_per_galaxy: int):
    """n_per_galaxy: 은하 하나당 입자 수. 총 입자 = n_per_galaxy * 2"""
    N = n_per_galaxy
    rng = np.random.default_rng(42)

    # Galaxy A
    x_A = rng.normal(10.0, 1.5, N);  y_A = rng.normal(0.0, 0.15, N);  z_A = rng.normal(0.0, 1.5, N)
    vx_A = np.full(N, -0.5);         vy_A = np.zeros(N);               vz_A = np.zeros(N)
    m_A  = np.ones(N)

    # Galaxy B
    x_B = rng.normal(-10.0, 1.5, N); y_B = rng.normal(0.0, 0.15, N);  z_B = rng.normal(0.0, 1.5, N)
    vx_B = np.full(N,  0.5);         vy_B = np.zeros(N);               vz_B = np.zeros(N)
    m_B  = np.ones(N)

    gal_A = np.column_stack((x_A, y_A, z_A, vx_A, vy_A, vz_A, m_A))
    gal_B = np.column_stack((x_B, y_B, z_B, vx_B, vy_B, vz_B, m_B))
    total = np.vstack((gal_A, gal_B))

    total_N = N * 2
    header  = np.array([total_N], dtype=np.int32).view(np.float32)
    data    = np.concatenate((header, total.flatten().astype(np.float32)))
    data.tofile(BIN_FILE)
    print(f"  [데이터 생성] 총 입자 수: {total_N}")


# ── 바이너리 실행 & 시간 파싱 ────────────────────
def run_and_parse(binary: str, workdir: str) -> dict:
    """바이너리 실행 후 '실행 시간'과 '성능' 파싱"""
    try:
        result = subprocess.run(
            [binary],
            capture_output=True, text=True,
            cwd=workdir, timeout=600
        )
        out = result.stdout
        elapsed_match = re.search(r"실행 시간[:：]\s*([\d.eE+\-]+)", out)
        gflops_match  = re.search(r"성능[:：]\s*([\d.eE+\-]+)", out)

        elapsed = float(elapsed_match.group(1)) if elapsed_match else None
        gflops  = float(gflops_match.group(1))  if gflops_match  else None
        return {"elapsed": elapsed, "gflops": gflops, "stdout": out}
    except subprocess.TimeoutExpired:
        print(f"    !! {binary} timeout (N이 너무 큼)")
        return {"elapsed": None, "gflops": None, "stdout": ""}
    except FileNotFoundError:
        print(f"    !! {binary} 를 찾을 수 없음. 빌드 확인 필요")
        return {"elapsed": None, "gflops": None, "stdout": ""}


# ── 메인 벤치마크 루프 ───────────────────────────
def main():
    cpu_times, gpu_times = [], []
    cpu_gflops, gpu_gflops = [], []
    total_ns = []

    for n in N_LIST:
        total_n = n * 2
        print(f"\n━━━ N={total_n} (은하당 {n}개) ━━━")

        generate_data(n)

        print("  [CPU] 실행 중...")
        # cwd를 cpu/ 로 설정 → 바이너리 내부 fopen("../data/gal_data.bin") 경로 정상 동작
        cpu_res = run_and_parse(os.path.abspath(CPU_BIN), "./cpu")
        cpu_t = cpu_res["elapsed"]
        print(f"  [CPU] 시간: {cpu_t:.3f}s" if cpu_t else "  [CPU] 실패")

        print("  [GPU] 실행 중...")
        # cwd를 gpu/ 로 설정 → 바이너리 내부 fopen("../data/gal_data.bin") 경로 정상 동작
        gpu_res = run_and_parse(os.path.abspath(GPU_BIN), "./gpu")
        gpu_t  = gpu_res["elapsed"]
        gpu_gf = gpu_res["gflops"]
        print(f"  [GPU] 시간: {gpu_t:.3f}s  성능: {gpu_gf:.2f} GFLOPS" if gpu_t else "  [GPU] 실패")

        # CPU GFLOPS 계산 (CPU 바이너리는 출력 없으므로 직접 계산)
        STEPS = 1000
        if cpu_t:
            cpu_gf = (STEPS * total_n * total_n * 20.0) / cpu_t / 1e9
        else:
            cpu_gf = None

        if cpu_t and gpu_t:
            speedup = cpu_t / gpu_t
            print(f"  [속도 향상] {speedup:.1f}x")

        total_ns.append(total_n)
        cpu_times.append(cpu_t)
        gpu_times.append(gpu_t)
        cpu_gflops.append(cpu_gf)
        gpu_gflops.append(gpu_gf)

    # ── 결과 저장 (CSV) ───────────────────────────
    with open("benchmark/results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "cpu_time_s", "gpu_time_s", "speedup", "cpu_gflops", "gpu_gflops"])
        for i, n in enumerate(total_ns):
            ct = cpu_times[i]; gt = gpu_times[i]
            speedup = (ct / gt) if (ct and gt) else ""
            writer.writerow([n, ct or "", gt or "", speedup, cpu_gflops[i] or "", gpu_gflops[i] or ""])
    print("\n결과 저장: benchmark/results.csv")

    # ── 그래프 ────────────────────────────────────
    plot_results(total_ns, cpu_times, gpu_times, cpu_gflops, gpu_gflops)


def plot_results(ns, cpu_times, gpu_times, cpu_gflops, gpu_gflops):
    valid = [(n, ct, gt, cg, gg)
             for n, ct, gt, cg, gg in zip(ns, cpu_times, gpu_times, cpu_gflops, gpu_gflops)
             if ct and gt]

    if not valid:
        print("유효한 결과 없음 — 그래프 생략")
        return

    ns_v, ct_v, gt_v, cg_v, gg_v = zip(*valid)
    speedups = [c / g for c, g in zip(ct_v, gt_v)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CPU vs GPU N-Body Performance Comparison", fontsize=15, fontweight="bold")

    # ── Graph 1: Execution Time ──
    ax = axes[0]
    ax.plot(ns_v, ct_v, "o-", color="#e74c3c", linewidth=2, markersize=6, label="CPU")
    ax.plot(ns_v, gt_v, "s-", color="#3498db", linewidth=2, markersize=6, label="GPU (Tiling + SoA)")
    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title("Execution Time Comparison")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Graph 2: Speedup ──
    ax = axes[1]
    ax.plot(ns_v, speedups, "D-", color="#2ecc71", linewidth=2, markersize=6, label="Speedup")
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Target: 50x")
    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("Speedup (CPU Time / GPU Time)")
    ax.set_title("GPU Speedup over CPU")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Graph 3: GFLOPS ──
    ax = axes[2]
    if any(cg_v):
        ax.plot(ns_v, cg_v, "o-", color="#e74c3c", linewidth=2, markersize=6, label="CPU")
    if any(gg_v):
        ax.plot(ns_v, [g for g in gg_v], "s-", color="#3498db", linewidth=2, markersize=6,
                label="GPU (Tiling + SoA)")
    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Throughput (GFLOPS)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "benchmark/performance_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Graph saved: {out_path}")


if __name__ == "__main__":
    main()
