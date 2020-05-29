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
# import scikit_image-0.12.1.dist-info
import argparse
from memory_profiler import profile
import ffht
import math
import time
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file
from sklearn_extra.kernel_approximation import Fastfood
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
import scipy
import numpy as np
from scipy.stats import chi
import sklearn
import matplotlib.pyplot as plt
from scipy._lib._util import _asarray_validated
from sklearn.utils import check_array, check_random_state
from extra_function import *
def main(name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # ax2 = ax1.twinx()
    '''
    read data and parameter initialize for the hadamard transform
    '''
    T_range = [1,2,4,8,16]
    marker = ['.','v','^','s','*','+']
    color = ['r','g','b','c','y','k']
    iter_acc_t = []
    iter_acc_b = []
    for T in T_range:
        for iter in range(3):

                x_train, y_train, x_test, y_test = get_data(name,FLAGS)
                FLAGS.T = T
                x, y = x_train, y_train
                class_number = len(np.unique(y))
                if class_number == 2:
                    class_number -= 1
                n_number, f_num = np.shape(x)
                d = 2 ** math.ceil(np.log2(f_num))
                G = np.random.randn(T * d)
                B = np.random.uniform(-1, 1, T * d)
                B[B > 0] = 1
                B[B < 0] = -1
                # PI_value = np.hstack
                PI_value = np.hstack([(i * d) + np.random.permutation(d) for i in range(T)])
                G_fro = G.reshape(T, d)
                s_i = chi.rvs(d, size=(T, d))
                S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1) ** (-1)).reshape(T, -1))
                S = S.reshape(1, -1)
                FLAGS.BATCHSIZE = n_number


                x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S,FLAGS))
                y_temp= label_processing(y,n_number,FLAGS)
                clf = LinearSVC()
                clf.fit(x_value, y)
                # print(clf.score(x_value,y))
                print('Training coordinate descent initiate from result of linear svm')
                iter_nonzero = []
                n_number, project_d = x_value.shape
                test_number, f_num = np.shape(x_test)
                FLAGS.BATCHSIZE = test_number
                test_x = np.asmatrix(hadamard(d, f_num, x_test, G, B, PI_value, S,FLAGS))

                test_y = label_processing(y_test, test_number,FLAGS)
                for w_type in range(2):
                    W_fcP = clf.coef_
                    W_fcP = np.asmatrix(np.sign(W_fcP.reshape(-1, class_number)))
                    if w_type==0:
                        W_fcP = optimization(x_value, y_temp, W_fcP, class_number)
                        clf2 = LinearSVC()
                        clf2.fit(np.dot(x_value, W_fcP), y)
                        iter_acc_t.append(clf2.score(np.dot(test_x, W_fcP), test_y))
                        # iter_acc_t.append(predict_acc(test_x, test_y, W_fcP, class_number))
                    else:
                        W_fcP = binary_optimization(x_value, y_temp, W_fcP, class_number)
                        clf2 = LinearSVC()
                        clf2.fit(np.dot(x_value, W_fcP), y)
                        iter_acc_b.append(clf2.score(np.dot(test_x, W_fcP), test_y))
                        # iter_acc_b.append(predict_acc(test_x, test_y, W_fcP, class_number))

    iter_acc_b = np.mean(np.array(iter_acc_b).reshape(len(T_range),-1),axis=1)
    iter_acc_t = np.mean(np.array(iter_acc_t).reshape(len(T_range), -1), axis=1)
    ax1.plot(T_range[:],iter_acc_b.reshape(1,-1)[0,:],linewidth=2,linestyle= '--',label = '{-1,1}',color =color[0],marker = marker[0],
             markersize=4)
    ax1.plot(T_range[:], iter_acc_t.reshape(1, -1)[0, :], linewidth=2, linestyle='--', label='{-1,0,1}', color=color[1],
             marker=marker[0],
             markersize=4)
    # ax2.plot(np.arange(6), iter_nonzero[:], linewidth = 2, linestyle = '--',color ='b', label = '# of non-zero' ,marker = '.',
    #          markersize = 10)T
    ax1.set_ylabel('acc')
    # ax2.set_ylabel('number of non-zero')
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.savefig('%s_%s.png' % (name, 'compare_w'))
    plt.show()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-T', type=int,
                        default=1,
                        help='number of the different hadamard random projection')
    parser.add_argument('-BATCHSIZE', type=int,
                        default=128,
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

    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name_space = ['CIFAR_10', 'Fashion-MNIST', 'mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    print(name)
    main(name)
