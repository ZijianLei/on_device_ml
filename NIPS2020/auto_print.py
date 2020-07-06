import os
import numpy as np
import time
from scipy.spatial.distance import hamming
import scipy
# from extra_function import *
from sklearn.kernel_approximation import RBFSampler
# T_number = [1,2,4,8,16,32]
# data = ['covtype']
# data = ['webspam']
data = ['usps']
# data = ['1','2']
for d in data:
        cmd = "nohup python  coordinate_descent.py -d_libsvm %s -method 1 &"%(d)
        cmd2 = "nohup python  coordinate_descent.py -d_libsvm %s -method 2 &" % (d)
        cmd3 = "nohup python  LSBC.py -d_libsvm %s  &" % (d)
        cmd4 = "nohup python  RKS_Fastfood.py -d_libsvm %s  &" % (d)
        # cmd = "nohup python  coordinate_descent.py -d_openml %s -method 1 &" % (d)
        # cmd2 = "nohup python  coordinate_descent.py -d_openml %s -method 2 &" % (d)
        # cmd3 = "nohup python  LSBC.py -d_openml %s  &" % (d)
        # cmd4 = "nohup python  RKS_Fastfood.py -d_openml %s &" % (d)
        os.system(cmd)
        os.system(cmd2)
        os.system(cmd3)
        os.system(cmd4)
# d = 1024
# T = 1
# print(2^8)
# X = np.random.uniform(-1, 1, (1,1024))
# norm1 = np.linalg.norm(X)
# print(norm1)
# h = scipy.linalg.hadamard(1024)
# norm2 = np.linalg.norm(np.dot(X,h))
# G = np.random.randn(T * d)
# B = np.random.uniform(-1, 1, T * d)
# B[B > 0] = 1
# B[B < 0] = -1
# # PI_value = np.hstack
# PI_value = np.hstack([(i * d) + np.random.permutation(d) for i in range(T)])
# G_fro = G.reshape(T, d)
# s_i = chi.rvs(d, size=(T, d))
# S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1)).reshape(T, -1))
# S = S.reshape(1, -1)
# x_ = X
# x_i = np.multiply(x_, S)
# for i in range(T):
#     ffht.fht(x_[i])
#
# x_transformed = np.multiply(x_, G)
# np.take(x_transformed, PI_value, axis=1, mode='wrap', out=x_)
#
# for i in range(T):
#     ffht.fht(x_[i])
# # print(np.linalg.norm(x_[:2],axis=1))
# # print(np.mean(x_[:,:2], axis=0))
# x_value = np.multiply(x_, B)/1024
# # x_temp = copy.deepcopy(x_value)*np.sqrt(2)
# # print(time.time()-start)
# norm3 = np.linalg.norm(x_value)
# rbf_feature = RBFSampler(gamma=1,random_state=1,n_components=3*d)
# sampler = rbf_feature.fit(X)
#
# x_value_rks = sampler.transform(X)
# norm4 = np.linalg.norm(x_value_rks)
# print(norm2,norm3,norm4,norm2/norm1,np.sqrt(1024))
