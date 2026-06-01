# gal_data.bin 파일 확인

import numpy as np

data = np.fromfile("gal_data.bin", dtype=np.float32)
print(data[0])
print(len(data))