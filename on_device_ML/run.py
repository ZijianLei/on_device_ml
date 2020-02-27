from scipy.linalg import hadamard
from sklearn_extra.utils._cyfht import fht as cyfht2
import time
from scipy.linalg import hadamard
import numpy as np
import ffht

A = np.random.random((10000,10,1024))
B = np.random.random((10000,10,1024))
start = time.time()
for i in range(A.shape[0]):
    for j in range(10):
        ffht.fht(A[i,j])
print(time.time()-start)
start = time.time()
for i in range(B.shape[0]):
    for j in range(10):
        cyfht2(B[i,j])
print(time.time()-start)
# print(A-B))
