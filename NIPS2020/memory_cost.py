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
import warnings
import ffht
import matplotlib.pyplot as plt
import numpy as np
from extra_function import *
import sklearn
from scipy._lib._util import _asarray_validated
from sklearn.utils import check_array, check_random_state
from sklearn.exceptions import DataConversionWarning
import time
import sys
from sklearn.kernel_approximation import *

def main(name):
    acc_1 = []
    acc_2 = []
    acc_3 = []
    print('start')
    '''
    read data and parameter initialize for the hadamard transform
    '''
    iteration = 1
    T_number = [1]
    # T_number = [8]
    sigma_number = [1]
    lambda_number = [1]
    # lambda_number = [0]
    acc_binary_1 = []
    acc_binary_2 = []
    acc_binary_3 = []
    non_zeros1 = []
    non_zeros2 = []
    non_zeros3 = []
    lamb = 1
    for T in T_number:
        for iter in range(iteration):
            x_train, y_train, x_test, y_test = get_data(name, FLAGS)
            for si in sigma_number:
                sigma = si
                print('Training Linear SVM')
                x, y = x_train, y_train
                FLAGS.T = T
                n_number, f_num = np.shape(x)
                d = 2 ** math.ceil(np.log2(f_num))
                FLAGS.BATCHSIZE = n_number
                # ==========Fastfood Based ================
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
                FLAGS.b = np.random.uniform(0, 2 * math.pi, d * T)
                FLAGS.t = np.random.uniform(-1, 1, d * T)
                y_temp = label_processing(y, n_number, FLAGS)
                temp = np.vstack((G,B,PI_value,FLAGS.b,FLAGS.t))

                # print(sys.getsizeof(temp)/1e3,temp.shape)
                # print((sys.getsizeof(G)+sys.getsizeof(B)+sys.getsizeof(PI_value)+sys.getsizeof(S)+sys.getsizeof( FLAGS.b)+sys.getsizeof( FLAGS.t))/1e3)
                class_number = len(np.unique(y))
                if class_number == 2:
                    class_number -= 1

                x_value = np.asmatrix(fastfood(d, f_num, x, G, B, PI_value, S, FLAGS, sigma))
                clf = LinearSVC(dual=False)
                clf.fit(x_value, y)
                print((sys.getsizeof(temp)+sys.getsizeof(clf.coef_)+sys.getsizeof(x_value[0]))/ 1e3, 'Fastfood')
                x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S, FLAGS, sigma))
                W_fcP = np.asmatrix(np.sign(np.random.randn(d * T, class_number)))
                W_fcP = optimization(x_value, y_temp, W_fcP, class_number, lamb)
                # print(sys.getsizeof(clf.coef_)+sys.getsizeof(x_value[0]),sys.getsizeof(np.array2string(W_fcP)))
                x_value = (x_value+1)/2
                print(((sys.getsizeof(temp)+np.count_nonzero(x_value[0])/8+np.count_nonzero(W_fcP)/8)/ 1e3, 'our proposd'))

                # ==========rks based ================+np.count_nonzero(W_fcP)/8+np.count_nonzero(x_value[0])/8)
                # clf2 = LinearSVC(dual = False)
                rbf_feature = RBFSampler(gamma=sigma, random_state=1, n_components=T * d)
                sampler = rbf_feature.fit(x)
                W_fcP = np.asmatrix(np.sign(np.random.randn(d * T, class_number)))
                x_rks = sampler.transform(x)
                x_value_rks = np.sign(sampler.transform(x))
                # clf2.fit(x_value_rks, y)
                y_temp = label_processing(y, n_number, FLAGS)
                # W_fcP = optimization(x_value_rks, y_temp, W_fcP, class_number)
                print((sys.getsizeof(clf.coef_)+sys.getsizeof(sampler.random_offset_)+sys.getsizeof(sampler.random_weights_) +sys.getsizeof(x_rks[0])) / 1e3, 'RKS')

                print((sys.getsizeof(clf.coef_)+sys.getsizeof(sampler.random_offset_)+sys.getsizeof(sampler.random_weights_) +sys.getsizeof(x_value_rks[0])) / 1e3, 'lsbc')
                # ==========random based ================
                V = np.random.randn(d, d * T)
                x_value = np.sign(np.dot(x,V))
                print((sys.getsizeof(V)+sys.getsizeof(clf2.coef_) +sys.getsizeof(x_value[0]))/ 1e3, 'SCBD')



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


    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name_space = ['CIFAR_10', 'Fashion-MNIST', 'mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    print(name)
    main(name)
