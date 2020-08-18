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

    acc_random = []
    acc_svm = []
    x_train, y_train, x_test, y_test = get_data(name, FLAGS)
    p = FLAGS.p
    for iter in range(5):

        x,y = x_train,y_train
        n_number, f_num = np.shape(x)
        d = 2 ** math.ceil(np.log2(f_num))

        FLAGS.p = p
        G = np.random.randn(p * d)
        B = np.random.uniform(-1, 1, p * d)
        B[B > 0] = 1
        B[B < 0] = -1
        # PI_value = np.hstack
        PI_value = np.hstack([(i * d) + np.random.permutation(d) for i in range(p)])
        G_fro = G.reshape(p, d)
        s_i = chi.rvs(d, size=(p, d))
        S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1)).reshape(p, -1))
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
        FLAGS.b = np.random.uniform(0, 2 * math.pi, d * p)
        FLAGS.t = np.random.uniform(-1, 1, d * p)
        sigma = FLAGS.s
        lamb = 0
        x, y = x_train, y_train
        FLAGS.p = p
        n_number, f_num = np.shape(x)
        FLAGS.BATCHSIZE = n_number
        y_temp = label_processing(y, n_number, FLAGS)

        x_value = np.asmatrix(feature_binarizing(d, f_num, x, G, B, PI_value, S, FLAGS, sigma))

        W_random = np.asmatrix(np.sign(np.random.randn(d * p, class_number)))

        # start = time.time()
        clf = LinearSVC(loss='hinge')
        clf.fit(x_value[:3000], y[:3000])
        alpha = np.linalg.norm(clf.coef_, ord=1, axis=1)

        W_svm = np.asmatrix(np.sign(clf.coef_)).T

        W_svm = optimization(x_value, y_temp, W_svm, class_number, alpha, lamb)

        alpha = np.sign(alpha)
        W_random = optimization(x_value, y_temp, W_random, class_number, alpha, lamb)
        # W_random,loss2 = compare_init_optimization(x_value, y_temp, W_random, class_number, lamb)
        # W_svm,loss3 = compare_init_optimization(x_value, y_temp, W_svm, class_number, lamb)
        x, y = x_test, y_test
        n_number, f_num = np.shape(x)
        FLAGS.data_number = n_number
        test_x = np.asmatrix(feature_binarizing(d, f_num, x, G, B, PI_value, S, FLAGS, sigma))
        acc_svm.append(predict_acc(x_value, y_train, test_x, y_test, W_svm))
        acc_random.append(predict_acc(x_value, y_train, test_x, y_test, W_random))

    plt.plot(np.arange(len(acc_random))+1, acc_random, linewidth=3, linestyle='--', label='random init',marker = 's',markersize = 12)
    plt.plot(np.arange(len(acc_svm))+1, acc_svm, linewidth=3, linestyle='--', label='init from svm',marker = 'v',markersize = 12)
    plt.legend(prop={'size': 15})
    # plt.xticks(np.arange(len(loss1)),fontsize=17)
    plt.yticks(fontsize=17)
    plt.ylabel('accuracy',fontsize=17)
    plt.xlabel('iteration',fontsize=17)
    x = range(1, 6, 1)
    plt.xticks(x,('1', '2', '3', '4', '5'), fontsize=17)
    plt.tight_layout()
    # plt.savefig('compare_init_%s.png' %name)
    plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
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
    parser.add_argument('-p', type=int,
                        default=8,
                        help='number of the different feature_binarizing random projection,\
                        the final dimension is p*d, d is the original dimension'
                             )
    parser.add_argument('-data_number', type=int,
                        default=None,
                        help='number of data')

    parser.add_argument('-s', type=float,
                        default=1,
                        help='sigma of random Gaussian distribution ')
    parser.add_argument('-l', type=float,
                        default=0,
                        help='the regularization parameter ')

    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name_space = ['CIFAR_10','Fashion-MNIST','mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    print(name)
    main(name)
