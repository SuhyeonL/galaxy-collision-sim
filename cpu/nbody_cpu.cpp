// cpu simulation
/*
1. .bin 파일 읽기
2. N-Body 계산
3. 위치 업데이트
4. 반복
*/

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono>

// 중력 상수 G
#define G 1.0f
// Epsilon: softening parameter
#define EPSILON 0.1f
// 시뮬레이션에서 한 스텝당 흐른 시간
#define DT 0.001f
// 계산 step
#define STEPS 100


int main() {
    // 1. 파일 읽기
    // 1.1 파일 열기
    FILE* fp = fopen("../data/gal_data.bin", "rb"); // rb = read binary
    if(fp == NULL) {
        std::cout << "file failed\n";
    }

    // 1.2 N 읽기
    float N;
    fread(&N, sizeof(float), 1, fp);
    // N check
    // std::cout << N << '\n';

    // 1.3 입자 데이터 읽기
    float* data = new float[N * 7];
    fread(data, sizeof(float), N * 7, fp);

    // 1.4 파일 닫기
    fclose(fp);

    std::cout << "입자 0 초기 위치: ";
    std::cout << data[0] << ' ' << data[1] << ' ' << data[2] << '\n';

    // 2. N-Body
    // N-Body 계산식으로 하나의 입자의 가속도를 구하는 것
    // 구해진 가속도를 가지고 속도와 위치를 업데이트 함
    float a, m, p_i[3], p_j[3], dx, dy, dz, dist;
    // 가속도
    float ax, ay, az;
    float* ax_arr = new float[N];
    float* ay_arr = new float[N];
    float* az_arr = new float[N];
    
    auto st = std::chrono::high_resolution_clock::now();

    for(int step = 0; step < STEPS; step++) {
        for(int i = 0; i < N; i++) { 
            ax = 0; ay = 0; az = 0; 

            // 가속도 계산
            for(int j = 0; j < N; j++) { 
                if(i == j) continue;
                m = data[j*7 + 6]; 
                for(int k = 0; k < 3; k++) { 
                    p_i[k] = data[i*7 + k]; 
                    p_j[k] = data[j*7 + k]; 
                } 
                dx = p_j[0] - p_i[0]; 
                dy = p_j[1] - p_i[1]; 
                dz = p_j[2] - p_i[2]; 
                dist = sqrt(dx*dx + dy*dy + dz*dz);
                
                // 가속도는 x, y, z 각각 따로 계산 
                ax += G * m * dx / pow(dist*dist + EPSILON*EPSILON, 1.5); 
                ay += G * m * dy / pow(dist*dist + EPSILON*EPSILON, 1.5);
                az += G * m * dz / pow(dist*dist + EPSILON*EPSILON, 1.5); 
            } 
            // 가속도 저장
            ax_arr[i] = ax;
            ay_arr[i] = ay;
            az_arr[i] = az;
        }
        // 속도 & 위치 업데이트
        // 속도 업데이트 후 위치 업데이트 해야함
        // 속도 업데이트, 속도 = 속도 + 가속도 * 시간간격
        for(int i = 0; i < N; i++) {
            data[i*7 + 3] += ax_arr[i]*DT;
            data[i*7 + 4] += ay_arr[i]*DT;
            data[i*7 + 5] += az_arr[i]*DT;
        }
        // 위치 업데이트, 위치 = 위치 + 업데이트된 속도 * 시간간격
        // x += vx * DT
        for(int i = 0; i < N; i++) {
            data[i*7 + 0] += data[i*7 + 3]*DT;
            data[i*7 + 1] += data[i*7 + 4]*DT;
            data[i*7 + 2] += data[i*7 + 5]*DT;
        }
    }

    auto ed = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(ed - st).count();

    std::cout << "입자 0 최종 위치: ";
    std::cout << data[0] << ' ' << data[1] << ' ' << data[2] << '\n';

    std::cout << "실행 시간: " << elapsed << '\n';


    delete [] ax_arr;
    delete [] ay_arr;
    delete [] az_arr;
    delete [] data;
}

