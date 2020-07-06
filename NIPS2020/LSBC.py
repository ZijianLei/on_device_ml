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
    T_number = [32]
    T_number = [1,2,4,8,16,32]
    sigma_number = [2 ** 9, 2 ** 6, 2 ** 3, 1, 2 ** (-3), 2 ** (-6), 2 ** (-9)]
    # lambda_number = [1e3, 1e2, 1e1, 1, 0]
    # lambda_number = [0]
    acc_binary_1 = []
    acc_binary_2 = []
    acc_binary_3 = []
    non_zeros1 = []
    non_zeros2 = []
    non_zeros3 = []
    for T in T_number:
        for iter in range(iteration):
            x_train, y_train, x_test, y_test = get_data(name, FLAGS)
            for si in sigma_number:
                sigma = si
                print('Training Linear SVM')
                x, y = x_train, y_train
                class_number = len(np.unique(y))
                if class_number == 2:
                    class_number -= 1
                FLAGS.T = T
                n_number, f_num = np.shape(x)
                d = 2 ** math.ceil(np.log2(f_num))
                FLAGS.BATCHSIZE = n_number
                clf2 = LinearSVC(dual = False)
                rbf_feature = RBFSampler(gamma=sigma, random_state=1, n_components=T * d)
                sampler = rbf_feature.fit(x)
                W_fcP = np.asmatrix(np.sign(np.random.randn(d * T, class_number)))
                x_value_rks = np.sign(sampler.transform(x))
                clf2.fit(x_value_rks, y)
                y_temp = label_processing(y, n_number, FLAGS)
                W_fcP = optimization(x_value_rks, y_temp, W_fcP, class_number)
                '''
                Start the test process
                '''
                x, y = x_test, y_test
                n_number, f_num = np.shape(x)
                FLAGS.BATCHSIZE = n_number
                test_x_rks = np.sign(sampler.transform(x))
                acc_binary_2.append(clf2.score(test_x_rks, y))
                acc_binary_3.append(predict_acc(x_value_rks,y_train,test_x_rks,y_test,W_fcP,class_number))
    acc_binary_2 = np.array(acc_binary_2)
    acc_binary_3 = np.array(acc_binary_3)
    np.save('%s_lsbc_8' % name, acc_binary_2)
    np.save('%s_lsbc_sign_8' % name, acc_binary_3)
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
