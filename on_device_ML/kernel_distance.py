from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#import scikit_image-0.12.1.dist-info
import argparse
from memory_profiler import profile
import  warnings
import ffht
import matplotlib.pyplot as plt
import numpy as np
from sklearn import *
from extra_function import  *
from sklearn.kernel_approximation import RBFSampler
import math
import sklearn
def main(name ):
    x_train, y_train, x_test, y_test = get_data(name,FLAGS)
    x, y = x_train, y_train
    n_number, f_num = np.shape(x)
    d = 2 ** math.ceil(np.log2(f_num))
    T = 8
    V = np.random.randn(f_num,d*T)
    kernel  = metrics.pairwise.rbf_kernel(x[:500])
    distance = euclidean_distances(x[:500])
    PI_value, G, B, S = fastfood_value(500,T,d,FLAGS)
    print('plot')
    # FLAGS.b = np.random.uniform(0, 2 * math.pi, d * T)
    # FLAGS.t = np.random.uniform(-1, 1, d * T)
    x_value = hadamard(d, f_num, x[:500], G, B, PI_value, S, FLAGS,sigma = 0.5*n_number) # the resukt of Fastfood-ocs
    distance2 = np.dot(x_value[:500],x_value[:500].T)/(d*T)
    # distance2 = dist.pairwise(x_value[:500])
    # print(distance2.shape)
    FLAGS.b  = None
    x_value2 =  hadamard(d, f_num, x[:500], G, B, PI_value, S, FLAGS)
    distance_sign = np.dot(x_value2,x_value2.T)/(d*T)
    # distance2 = np.dot(x_temp[:500], x_temp[:500].T)/np.max(np.dot(x_temp[:500], x_temp[:500].T))
    # plt.scatter(distance[:, :], distance2[:, :], zorder=15, s=1)
    plt.scatter(np.triu(distance[:,:]),np.triu(kernel[:,:]),zorder=30,s=5,label = r'rbf kernel $k(x,y)$') # the result of rbf-kernel
    rbf_feature = RBFSampler(gamma=1/d, random_state=1, n_components=f_num)
    x_rks =np.sign( rbf_feature.fit_transform(x[:500]))
    # x_rks = sklearn.preprocessing.normalize(x_rks,axis=1)
    distance3 = np.dot(x_rks,x_rks.T)/f_num
    x_sign = np.sign(np.dot(x[:500],V))
    distance4 = np.array(np.dot(x_sign[:500],x_sign[:500].T)/(d*T))
    # print(type(distance4),type(distance))
    plt.scatter(np.triu(distance[:, :]), np.triu(distance4[:,:]), zorder=15, s=5, label='sign(Vx)')
    plt.scatter(np.triu(distance[:,:]),np.triu(distance3[:,:]),zorder=15,s=5,label = r'LSBC')
    # plt.scatter(distance[:, :], distance_sign[:, :], zorder=5, s=5, label=r'sign(Vx)')

    plt.scatter(np.triu(distance[:, :]), np.triu(distance2[:, :]), zorder=20, s=5, label=r'Our Proposed $\frac{1}{n}(z_xz_y)$')

    # FLAGS.b = None
    # x_value = hadamard(d, f_num, x, G, B, PI_value, S, FLAGS,sigma = 1/n_number)
    # distance2 = np.dot(x_value[:500], x_value[:500].T) / (d * T)
    # plt.scatter(distance[:, :], distance2[:, :],zorder=10,s=1)
    # distance2 = np.dot(x_temp[:500], x_temp[:500].T) / (d * T)
    # plt.scatter(distance[:, :], distance2[:, :], zorder=15, linewidths=1)
    plt.legend()
    plt.ylabel('kernel value')
    plt.xlabel('euclidean distance')
    plt.savefig('distance.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser.add_argument('-T', type=int,
                        default=None,
                        help='number of the different hadamard2 random projection')
    parser.add_argument('-BATCHSIZE', type=int,
                        default=128,
                        help='number of data')
    parser.add_argument('-b', type=float,
                        default=None,
                        help='number of data')
    parser.add_argument('-t', type=float,
                        default=None,
                        help='number of data')

    parser.add_argument('-d_openml', type=int,
                        default=None,
                        help='data is download from openml\
                        available data:\
                        0:CIFAR_10,\
                        1:Fashion-MNIST,\
                        2:mnist_784'                           )
    '''
        available data:
        0:CIFAR_10,
        1:Fashion-MNIST,
        2:mnist_784
    '''
    parser.add_argument('-d_libsvm', type=str,
                        default=None,
                        help='using data from libsvm dataset with data is well preprocessed')
    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name_space = ['CIFAR_10','Fashion-MNIST','mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    main(name)
