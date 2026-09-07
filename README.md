# galaxy-collision-sim
Parallel Processing final project

# 프로젝트 소개
CUDA를 활용한 N-Body 시뮬레이션(운석 충돌) 프로젝트. CPU/GPU 구현을 비교하고 성능을 벤치마크함

## 폴더 구조
- cpu/ : CPU 기반 구현
- gpu/ : GPU(CUDA) 기반 구현
- data/ : 시뮬레이션 입력 데이터 생성
- benchmark/ : 성능 측정
- viz/ : 결과 시각화

## 실행 방법
### 1. 입자 데이터 생성
python data/generate_data.py
# gal_data.bin 파일이 생성됩니다.

(선택) 생성된 데이터 확인:
python data/ch.py

### 2. CPU 버전 컴파일 및 실행
g++ cpu/nbody_cpu.cpp -o nbody_cpu
./nbody_cpu

### 3. GPU 버전 컴파일 및 실행
nvcc gpu/nbody-gpu.cu -o nbody_gpu
./nbody_gpu

### 4. 성능 벤치마크
python benchmark/benchmark.py

### 5. 결과 시각화
python viz/viz.py
