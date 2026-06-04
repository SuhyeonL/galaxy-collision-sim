// gpu simulation
/*
1. .bin 파일 읽기
2. GPU 메모리로 데이터 복사
3. CUDA 커널로 N-Body 계산 (병렬)
4. 위치/속도 업데이트 (병렬)
5. 반복
6. viz_data.bin 저장
*/

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>
#include <vector>
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
// 커널 1: 가속도 계산
//   각 스레드가 입자 i 하나를 담당
//   모든 j에 대해 중력 가속도를 누적
// ─────────────────────────────────────────────
__global__ void compute_acceleration(
    const float* data,   // [N * 7]: x,y,z, vx,vy,vz, m
    float* ax_arr,
    float* ay_arr,
    float* az_arr,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float px_i = data[i*7 + 0];
    float py_i = data[i*7 + 1];
    float pz_i = data[i*7 + 2];

    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    for (int j = 0; j < N; j++) {
        if (i == j) continue;

        float dx = data[j*7 + 0] - px_i;
        float dy = data[j*7 + 1] - py_i;
        float dz = data[j*7 + 2] - pz_i;
        float m  = data[j*7 + 6];

        float dist2 = dx*dx + dy*dy + dz*dz + EPSILON*EPSILON;
        float inv   = rsqrtf(dist2);          // 1/sqrt(dist2)
        float inv3  = inv * inv * inv;        // 1/dist^3 (softened)
        float coeff = G * m * inv3;

        ax += coeff * dx;
        ay += coeff * dy;
        az += coeff * dz;
    }

    ax_arr[i] = ax;
    ay_arr[i] = ay;
    az_arr[i] = az;
}


// ─────────────────────────────────────────────
// 커널 2: 속도 & 위치 업데이트
//   각 스레드가 입자 i 하나를 담당
// ─────────────────────────────────────────────
__global__ void update_velocity_position(
    float* data,
    const float* ax_arr,
    const float* ay_arr,
    const float* az_arr,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // 속도 업데이트: v += a * dt
    data[i*7 + 3] += ax_arr[i] * DT;
    data[i*7 + 4] += ay_arr[i] * DT;
    data[i*7 + 5] += az_arr[i] * DT;

    // 위치 업데이트: x += v * dt
    data[i*7 + 0] += data[i*7 + 3] * DT;
    data[i*7 + 1] += data[i*7 + 4] * DT;
    data[i*7 + 2] += data[i*7 + 5] * DT;
}


// ─────────────────────────────────────────────
// 커널 3: viz_data 저장 (매 스텝 위치 기록)
// ─────────────────────────────────────────────
__global__ void save_positions(
    const float* data,
    float* viz_step,   // [N*3]: 현재 스텝의 위치 저장 공간
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    viz_step[i*3 + 0] = data[i*7 + 0];
    viz_step[i*3 + 1] = data[i*7 + 1];
    viz_step[i*3 + 2] = data[i*7 + 2];
}


int main() {
    // ── 1. 파일 읽기 ──────────────────────────────
    FILE* fp = fopen("../data/gal_data.bin", "rb");
    if (fp == NULL) {
        std::cout << "file open failed\n";
        return 1;
    }

    int N;
    fread(&N, sizeof(int), 1, fp);

    float* h_data = new float[N * 7];
    fread(h_data, sizeof(float), N * 7, fp);
    fclose(fp);

    std::cout << "입자 수: " << N << '\n';
    std::cout << "입자 0 초기 위치: "
              << h_data[0] << ' ' << h_data[1] << ' ' << h_data[2] << '\n';

    // ── 2. GPU 메모리 할당 & 복사 ────────────────
    float *d_data, *d_ax, *d_ay, *d_az, *d_viz;

    CUDA_CHECK(cudaMalloc(&d_data, N * 7 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ax,   N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ay,   N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_az,   N     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_viz,  (long long)STEPS * N * 3 * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * 7 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ── 3. 시뮬레이션 루프 ────────────────────────
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto st = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < STEPS; step++) {
        // 가속도 계산
        compute_acceleration<<<blocks, BLOCK_SIZE>>>(d_data, d_ax, d_ay, d_az, N);

        // 속도 & 위치 업데이트
        update_velocity_position<<<blocks, BLOCK_SIZE>>>(d_data, d_ax, d_ay, d_az, N);

        // 이번 스텝 위치 viz 버퍼에 저장
        float* viz_step_ptr = d_viz + (long long)step * N * 3;
        save_positions<<<blocks, BLOCK_SIZE>>>(d_data, viz_step_ptr, N);
    }

    // 모든 커널 완료 대기
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ed = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(ed - st).count();

    // ── 4. 결과 CPU로 복사 ────────────────────────
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * 7 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "입자 0 최종 위치: "
              << h_data[0] << ' ' << h_data[1] << ' ' << h_data[2] << '\n';
    std::cout << "실행 시간: " << elapsed << "s\n";

    // ── 5. viz_data.bin 저장 ──────────────────────
    long long viz_total = (long long)STEPS * N * 3;
    float* h_viz = new float[viz_total];
    CUDA_CHECK(cudaMemcpy(h_viz, d_viz, viz_total * sizeof(float),
                          cudaMemcpyDeviceToHost));

    FILE* out = fopen("viz_data.bin", "wb");
    int steps = STEPS;
    fwrite(&steps,  sizeof(int), 1, out);
    fwrite(&N,      sizeof(int), 1, out);
    fwrite(h_viz, sizeof(float), viz_total, out);
    fclose(out);

    std::cout << "viz_data.bin 저장 완료\n";

    // ── 6. 메모리 해제 ─────────────────────────────
    cudaFree(d_data);
    cudaFree(d_ax);
    cudaFree(d_ay);
    cudaFree(d_az);
    cudaFree(d_viz);
    delete[] h_data;
    delete[] h_viz;

    return 0;
}
