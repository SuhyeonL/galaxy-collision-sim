// cpu simulation
/*
1. .bin 파일 읽기
2. N-Body 계산
3. 위치 업데이트
4. 반복
*/

#include <stdio.h>
#include <cmath>

int main() {
    // 1. 파일 읽기
    // 1.1 파일 열기
    FILE* fp = fopen("../data/gal_data.bin", "rb"); // rb = read binary

    // 1.2 N 읽기
    int N;
    fread(&N, sizeof(int), 1, fp);

    // 1.3 입자 데이터 읽기
    float* data = new float[N * 7];
    fread(data, sizeof(float), N * 7, fp);

    // 1.4 파일 닫기
    fclose(fp);


    // 2. N-Body
    // N-Body 계산식으로 하나의 입자의 가속도를 구하는 것
    // 구해진 가속도를 가지고 속도와 위치를 업데이트 함
    float a, G, m, p_i[3], p_j[3], dx, dy, dz, dist, epsilon;
    float ax=0, ay=0, az=0;
    
    for(int i = 0; i < N; i++) {
        ax = 0;
        ay = 0;
        az = 0;
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
            ax += G * m * dx / pow(dist*dist + epsilon*epsilon, 1.5);
            ay += G * m * dy / pow(dist*dist + epsilon*epsilon, 1.5);
            az += G * m * dz / pow(dist*dist + epsilon*epsilon, 1.5);

            // 속도랑 위치 업데이트 구현해야함

        }
    }



    delete [] data;
}

