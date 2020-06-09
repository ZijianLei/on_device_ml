# Copyright 2011 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

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
from extra_function import *
import sklearn
from scipy._lib._util import _asarray_validated
from sklearn.utils import check_array, check_random_state
from sklearn.exceptions import DataConversionWarning
import time
import matplotlib.pyplot as plt
def main(name ):
    acc_1 = []
    acc_2 = []
    acc_3 = []
    print('start')
    '''
    read data and parameter initialize for the hadamard transform
    '''
    iteration = 1
    # T_number = [1, 2, 4, 8, 16,32]
    T_number = [8]
    # sigma_number = [2 ** 9, 2 ** 6, 2 ** 3, 1, 2 ** (-3), 2 ** (-6), 2 ** (-9)]
    # sigma_number = [  2**5,2**(4),2 ** 3,4,2 ,1,2**(-1),2**(-2),2 ** (-3),2**(-4),2**(-5)]
    # lambda_number = [1e3,1e2,1e1,1,0]
    lambda_number = [0]
    sigma_number = [0.5]
    acc_binary_1 = []
    acc_binary_1_sign = []
    acc_binary_2_linear = []
    acc_binary_2 = []
    acc_binary_3 = []
    non_zeros1 = []
    non_zeros2 = []
    non_zeros3 = []
    method = FLAGS.method
    hinge_loss2 = []
    for T in T_number:
        x_train, y_train, x_test, y_test = get_data(name, FLAGS)
        x,y = x_train,y_train
        n_number, f_num = np.shape(x)
        d = 2 ** math.ceil(np.log2(f_num))

        FLAGS.T = T
        G = np.random.randn(T * d)
        B = np.random.uniform(-1, 1, T * d)
        B[B > 0] = 1
        B[B < 0] = -1
        # PI_value = np.hstack
        PI_value = np.hstack([(i * d) + np.random.permutation(d) for i in range(T)])
        G_fro = G.reshape(T, d)
        s_i = chi.rvs(d, size=(T, d))
        S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1)).reshape(T, -1))
        S = S.reshape(1, -1)
        FLAGS.BATCHSIZE = n_number
        class_number = len(np.unique(y))
        if class_number == 2:
            class_number -= 1
        '''
        Start to prossssscessing the label
        '''
        FLAGS.b = None
        FLAGS.t = None
        # if method !=0:
        FLAGS.b = np.random.uniform(0, 2 * math.pi, d * T)
        FLAGS.t = np.random.uniform(-1, 1, d * T)
        for si in sigma_number:
            sigma = si
            for lamb in lambda_number:
                x, y = x_train, y_train
                FLAGS.T = T
                n_number, f_num = np.shape(x)
                FLAGS.BATCHSIZE = n_number
                y_temp = label_processing(y, n_number, FLAGS)
                W_fcP = np.asmatrix(np.sign(np.zeros((d *T,class_number))))

                x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S, FLAGS, sigma))
                for c in range(class_number):
                    index_x = np.where(y_temp[:, c] == 1)
                    W_fcP[:, c] = np.asmatrix(np.sign(np.sum(x_value[index_x])))
                W_fcP2 = np.asmatrix(np.sign(np.random.randn(d * T, class_number)))
                clf = LinearSVC()
                clf.fit(x_value, y)
                # print(clf.coef_.shape)
                # exit()
                W_fcP3 = np.asmatrix(np.sign(clf.coef_)).T
                W_fcP4 = np.asmatrix(clf.coef_).T
                W_fcP,loss1 = compare_init_optimization(x_value, y_temp, W_fcP, class_number, lamb)
                W_fcP2,loss2 = compare_init_optimization(x_value, y_temp, W_fcP2, class_number, lamb)
                W_fcP3,loss3 = compare_init_optimization(x_value, y_temp, W_fcP3, class_number, lamb)
                # W_fcP4,loss4 = compare_init_optimization(x_value, y_temp, W_fcP4, class_number, lamb)
                # plt.plot(np.arange(len(loss1)),loss1,linewidth = 2,linestyle= '--',label = 'zero init')
                plt.plot(np.arange(len(loss2)), loss2, linewidth=3, linestyle='--', label='random init',marker = 's',markersize = 12)
                plt.plot(np.arange(len(loss3)), loss3, linewidth=3, linestyle='--', label='init from svm',marker = 'v',markersize = 12)
                # plt.plot(np.arange(len(loss4)), loss4, linewidth=2, linestyle='--', label='sign init from svm')
                plt.legend(prop={'size': 15})
                plt.xticks(np.arange(len(loss2)),fontsize=17)
                plt.yticks(fontsize=17)
                plt.ylabel('objective value',fontsize=17)
                plt.xlabel('iteration',fontsize=17)
                plt.tight_layout()
                plt.savefig('compare_init.png')
                plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser.add_argument('-T', type=int,
                        default=None,
                        help='number of the different hadamard random projection')
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
                        2:mnist_784'
                             )
    '''
        available data:
        0:CIFAR_10,
        1:Fashion-MNIST,
        2:mnist_784
    '''
    parser.add_argument('-d_libsvm', type=str,
                        default=None,
                        help='using data from libsvm dataset with data is well preprocessed')
    parser.add_argument('-method', type=int,
                        default=2,
                        help='using data from libsvm dataset with data is well preprocessed')

    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name_space = ['CIFAR_10','Fashion-MNIST','mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    print(name)
    main(name)
