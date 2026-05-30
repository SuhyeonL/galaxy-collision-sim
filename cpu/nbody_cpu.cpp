// cpu simulation
/*
1. .bin 파일 읽기
2. N-Body 계산
3. 위치 업데이트
4. 반복
*/

#include <stdio.h>

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
}

