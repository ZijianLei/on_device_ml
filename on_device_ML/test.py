import sklearn.metrics
import numpy as np
import time
p =2048
c = np.array([10,1,1,10,10])
d = np.array([256,54,254,780,780])
result = (p*(32*3+1+c))/(8*1024)
print("&"+"kb & ".join(str(int(i)) for i in result)+"kb \\")

exit()
A = np.asmatrix(np.random.randn(2**19,2**6))
np.sign(A,out = A)
n = 2**19
W = np.asmatrix(np.random.randn(2**6,1))
np.sign(W,out = W)
print(type(W))
y = np.random.randn(2**19,1)
print(y.shape)
np.sign(y,out = y)
print(y.shape)
start = time.time()
init = A.dot(W)
hinge_loss = sklearn.metrics.hinge_loss(y, init)
print(hinge_loss)
for i in range(2**6):
    w_i = W[i]
    b = np.multiply(A[:, i],w_i)
    init = init - 2*np.multiply(A[:, i],w_i)
    hinge_loss = sklearn.metrics.hinge_loss(y, init)
print(hinge_loss,time.time()-start)

start = time.time()
a =  1- np.multiply(y,A.dot(W))
hinge_loss2 = np.sum(np.clip(a,0,None))/n
print(hinge_loss2)
for i in range(2**6):
    w_i = W[i]
    a += 2*np.multiply(y,A[:,i]*w_i)
    hinge_loss2 = np.sum(np.clip(a,0,None))/n
print(hinge_loss2,time.time()-start)