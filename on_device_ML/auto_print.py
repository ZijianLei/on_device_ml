import os
T_number = [1,2,4,8,16,32]
data = ['webspam','covtype']
for d in data:
    for T in T_number:
        cmd = "nohup python -u coordinate_descent.py -d_libsvm %s -T %d >>%s.txt &"%(d,T,d)
        os.system(cmd)