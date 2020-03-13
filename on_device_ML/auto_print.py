import os
T_number = [8,16]
data = 2
for T in T_number:
    cmd = "nohup python -u coordinate_descent.py -d_openml %d -T %d >>%d.txt &"%(data,T,data)
    os.system(cmd)