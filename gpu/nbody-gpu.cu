// gpu simulation
/*
1. .bin 파일 읽기 (AoS 포맷)
2. AoS → SoA 변환 후 GPU 메모리로 복사
3. CUDA 커널로 N-Body 계산 (병렬 + Shared Memory Tiling)
4. 위치/속도 업데이트 (병렬)
5. 반복
6. viz_data.bin 저장
*/

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// 중력 상수 G
#define G 1.0f
// Epsilon: softening parameter
#define EPSILON 0.1f
// 시뮬레이션에서 한 스텝당 흐른 시간
#define DT 0.005f
// 계산 step
#define STEPS 1000
// 블록당 스레드 수 (보통 128 or 256 사용)
#define BLOCK_SIZE 256


// CUDA 에러 체크 매크로
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d -> %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


// ─────────────────────────────────────────────
// 커널 1: 가속도 계산 (Shared Memory Tiling + SoA)
//
//   [SoA 최적화 원리]
//   - AoS: data[i*7+0], data[i*7+1], ...
//     → 스레드 0은 0번지, 스레드 1은 7번지 읽음 (stride=7, not coalesced)
//   - SoA: x[i], y[i], z[i], vx[i], ...
//     → 스레드 0은 x[0], 스레드 1은 x[1] (연속 접근, coalesced) ✅
//
//   [Shared Memory Tiling 원리]
//   - 전체 입자를 BLOCK_SIZE 크기 타일로 분할
//   - 블록 내 스레드들이 협력해 타일을 shared memory에 로드
//   - global memory read: N² → N²/BLOCK_SIZE
// ─────────────────────────────────────────────

// shared memory 타일에 저장할 입자 구조체 (x, y, z, m)
struct Particle4 {
    float x, y, z, m;
};

__global__ void compute_acceleration(
    const float* __restrict__ px,  // SoA: x 좌표 배열 [N]
    const float* __restrict__ py,  // SoA: y 좌표 배열 [N]
    const float* __restrict__ pz,  // SoA: z 좌표 배열 [N]
    const float* __restrict__ pm,  // SoA: 질량 배열   [N]
    float* ax_arr,
    float* ay_arr,
    float* az_arr,
    int N
) {
    // shared memory 타일: 한 번에 BLOCK_SIZE개 입자를 캐시
    __shared__ Particle4 s_tile[BLOCK_SIZE];

    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // i >= N인 스레드도 타일 로딩에는 참여해야 하므로
    // 위치는 미리 읽고, 가속도 누적만 조건부로 수행
    float px_i = (i < N) ? px[i] : 0.0f;
    float py_i = (i < N) ? py[i] : 0.0f;
    float pz_i = (i < N) ? pz[i] : 0.0f;

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    // 타일 단위로 순회
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        // ── 단계 1: 타일을 shared memory에 협력 로드 ──
        // SoA 배열에서 읽으므로 같은 배열 내 연속 주소 접근 → coalesced
        int j = t * BLOCK_SIZE + tx;
        if (j < N) {
            s_tile[tx].x = px[j];
            s_tile[tx].y = py[j];
            s_tile[tx].z = pz[j];
            s_tile[tx].m = pm[j];
        } else {
            // 패딩: 타일 끝이 N을 넘으면 질량을 0으로 → 가속도 기여 없음
            s_tile[tx] = {0.0f, 0.0f, 0.0f, 0.0f};
        }

        // ── 단계 2: 타일 로딩 완료 동기화 ──
        __syncthreads();

        // ── 단계 3: shared memory의 타일로 가속도 누적 ──
        if (i < N) {
            for (int k = 0; k < BLOCK_SIZE; k++) {
                int global_j = t * BLOCK_SIZE + k;
                if (global_j == i) continue;   // 자기 자신 제외

                float dx = s_tile[k].x - px_i;
                float dy = s_tile[k].y - py_i;
                float dz = s_tile[k].z - pz_i;
                float m  = s_tile[k].m;

                float dist2 = dx*dx + dy*dy + dz*dz + EPSILON*EPSILON;
                float inv   = rsqrtf(dist2);     // 1/sqrt(dist2)
                float inv3  = inv * inv * inv;   // 1/dist^3 (softened)
                float coeff = G * m * inv3;

                ax += coeff * dx;
                ay += coeff * dy;
                az += coeff * dz;
            }
        }

        // ── 단계 4: 다음 타일 로딩 전 동기화 ──
        __syncthreads();
    }

    if (i < N) {
        ax_arr[i] = ax;
        ay_arr[i] = ay;
        az_arr[i] = az;
    }
}


// ─────────────────────────────────────────────
// 커널 2: 속도 & 위치 업데이트 (SoA)
//   각 스레드가 입자 i 하나를 담당
//   SoA: vx[i], vy[i], vz[i], px[i], ... 접근 → coalesced
// ─────────────────────────────────────────────
__global__ void update_velocity_position(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* ax_arr,
    const float* ay_arr,
    const float* az_arr,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 속도 업데이트: v += a * dt
    vx[i] += ax_arr[i] * DT;
    vy[i] += ay_arr[i] * DT;
    vz[i] += az_arr[i] * DT;

    // 위치 업데이트: x += v * dt
    px[i] += vx[i] * DT;
    py[i] += vy[i] * DT;
    pz[i] += vz[i] * DT;
}


// ─────────────────────────────────────────────
// 커널 3: viz_data 저장 (매 스텝 위치 기록, SoA)
// ─────────────────────────────────────────────
__global__ void save_positions(
    const float* px,
    const float* py,
    const float* pz,
    float* viz_step,   // [N*3]: 현재 스텝의 위치 저장 공간
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    viz_step[i*3 + 0] = px[i];
    viz_step[i*3 + 1] = py[i];
    viz_step[i*3 + 2] = pz[i];
}


int main() {
    // ── 1. 파일 읽기 (AoS 포맷 그대로) ──────────
    FILE* fp = fopen("../data/gal_data.bin", "rb");
    if (fp == NULL) {
        std::cout << "file open failed\n";
        return 1;
    }

    int N;
    fread(&N, sizeof(int), 1, fp);

    // AoS 버퍼: [x,y,z,vx,vy,vz,m] * N
    float* h_aos = new float[N * 7];
    fread(h_aos, sizeof(float), N * 7, fp);
    fclose(fp);

    std::cout << "입자 수: " << N << '\n';
    std::cout << "입자 0 초기 위치: "
              << h_aos[0] << ' ' << h_aos[1] << ' ' << h_aos[2] << '\n';

    // ── 2. AoS → SoA 변환 ────────────────────────
    // GPU 커널이 coalesced access 하도록 분리된 배열로 변환
    float* h_px = new float[N];
    float* h_py = new float[N];
    float* h_pz = new float[N];
    float* h_vx = new float[N];
    float* h_vy = new float[N];
    float* h_vz = new float[N];
    float* h_m  = new float[N];

    for (int i = 0; i < N; i++) {
        h_px[i] = h_aos[i*7 + 0];
        h_py[i] = h_aos[i*7 + 1];
        h_pz[i] = h_aos[i*7 + 2];
        h_vx[i] = h_aos[i*7 + 3];
        h_vy[i] = h_aos[i*7 + 4];
        h_vz[i] = h_aos[i*7 + 5];
        h_m[i]  = h_aos[i*7 + 6];
    }
    delete[] h_aos;

    // ── 3. GPU 메모리 할당 & 복사 (SoA) ─────────
    float *d_px, *d_py, *d_pz;
    float *d_vx, *d_vy, *d_vz;
    float *d_m;
    float *d_ax, *d_ay, *d_az;
    float *d_viz;

    CUDA_CHECK(cudaMalloc(&d_px, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vy, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_vz, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ax, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ay, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_az, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viz, (long long)STEPS * N * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_px, h_px, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m,  h_m,  N * sizeof(float), cudaMemcpyHostToDevice));

    // ── 4. 시뮬레이션 루프 ────────────────────────
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto st = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; step++) {
        // 가속도 계산 (Shared Memory Tiling + SoA)
        compute_acceleration<<<blocks, BLOCK_SIZE>>>(
            d_px, d_py, d_pz, d_m,
            d_ax, d_ay, d_az, N
        );

        // 속도 & 위치 업데이트
        update_velocity_position<<<blocks, BLOCK_SIZE>>>(
            d_px, d_py, d_pz,
            d_vx, d_vy, d_vz,
            d_ax, d_ay, d_az, N
        );

        // 이번 스텝 위치 viz 버퍼에 저장
        float* viz_step_ptr = d_viz + (long long)step * N * 3;
        save_positions<<<blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz, viz_step_ptr, N);
    }

    // 모든 커널 완료 대기
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ed = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(ed - st).count();

    // ── 5. 결과 CPU로 복사 ────────────────────────
    CUDA_CHECK(cudaMemcpy(h_px, d_px, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_py, d_py, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pz, d_pz, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "입자 0 최종 위치: "
              << h_px[0] << ' ' << h_py[0] << ' ' << h_pz[0] << '\n';
    std::cout << "실행 시간: " << elapsed << "s\n";

    // GFLOPS 출력
    // N-Body 1스텝당 연산량: N * N * 20 flops (dx,dy,dz,dist2,rsqrt,inv3,coeff,ax,ay,az)
    double flops = (double)STEPS * (double)N * (double)N * 20.0;
    double gflops = flops / elapsed / 1e9;
    std::cout << "성능: " << gflops << " GFLOPS\n";

    // ── 6. viz_data.bin 저장 ──────────────────────
    long long viz_total = (long long)STEPS * N * 3;
    float* h_viz = new float[viz_total];
    CUDA_CHECK(cudaMemcpy(h_viz, d_viz, viz_total * sizeof(float),
                          cudaMemcpyDeviceToHost));

    FILE* out = fopen("viz_data.bin", "wb");
    int steps = STEPS;
    fwrite(&steps, sizeof(int), 1, out);
    fwrite(&N,     sizeof(int), 1, out);
    fwrite(h_viz, sizeof(float), viz_total, out);
    fclose(out);

    std::cout << "viz_data.bin 저장 완료\n";

    // ── 7. 메모리 해제 ─────────────────────────────
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_m);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    cudaFree(d_viz);
    delete[] h_px; delete[] h_py; delete[] h_pz;
    delete[] h_vx; delete[] h_vy; delete[] h_vz;
    delete[] h_m;
    delete[] h_viz;

    return 0;
}
