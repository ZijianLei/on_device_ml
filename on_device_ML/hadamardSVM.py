# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
#import scikit_image-0.12.3.dist-info
import argparse
from memory_profiler import profile
import ffht
import math
import time
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file
from sklearn.svm import  SVC,LinearSVC
import scipy
import numpy as np
from scipy.stats import chi
import sklearn
from scipy._lib._util import _asarray_validated


def get_data(name):
    if FLAGS.d_openml != None:
        if name == 'CIFAR_10':
            x, y = sklearn.datasets.fetch_openml(name=name, return_X_y=True)
            x = x/255
            x_train, x_test = x[:50000], x[50000:]
            y_train, y_test = y[:50000], y[50000:]

        else:
            x,y= sklearn.datasets.fetch_openml(name = name,return_X_y= True)
            x = x / 255
            x_train,x_test = x[:60000],x[60000:]
            y_train,y_test = y[:60000],y[60000:]
    else:
        x_train,y_train = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" %(name,'train'))
        x_test,y_test = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'test'))
        x_train = x_train.todense()
        x_test = x_test.todense()
    return x_train,y_train,x_test,y_test


# the fast implement of using hadamard transform to apporximate the gaussian random projection
def hadamard(d,f_num,batch,G,B,PI_value,S):
    T = FLAGS.T
    x_ = batch
    x_ = np.pad(x_, ((0,0),(0, d - f_num)), 'constant', constant_values=(0, 0))  # x.shape [batch,d], pad to d = 2^p
    x_ = np.tile(x_, (1, T))  #
    x_i = np.multiply(x_, S)
    x_ = x_i.reshape(FLAGS.BATCHSIZE, T, d)
    for i in range(x_.shape[0]):
        for j in range(T):
            ffht.fht(x_[i,j])
    x_transformed = np.multiply(x_.reshape(FLAGS.BATCHSIZE, d * T), G)
    x_transformed = np.reshape(x_transformed, (FLAGS.BATCHSIZE, T, d))
    x_ = x_transformed
    for i in range(x_.shape[0]):
        for j in range(T):
            ffht.fht(x_[i, j])
    x_ = x_.reshape(FLAGS.BATCHSIZE, T * d)
    x_value = np.multiply(x_, B)
    x_value = np.sign(x_value)
    return x_value


def main(name ):
    #read the dataset and initialize the parameter for hadamard transform
    x_train, y_train, x_test, y_test= get_data(name)
    x,y = x_train,y_train
    n_number, f_num = np.shape(x)
    FLAGS.BATCHSIZE = n_number
    d = 2 ** math.ceil(np.log2(f_num))
    T = FLAGS.T
    G = np.random.randn(T * d)
    B = np.random.uniform(-1, 1, T * d)
    B[B > 0] = 1
    B[B < 0] = -1
    PI_value = np.random.permutation(d)
    G_fro = G.reshape(T, d)
    s_i = chi.rvs(d, size=(T, d))
    S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1) ** (-0.5)).reshape(T, -1))
    S = S.reshape(1, -1)
    print(np.unique(y))

    class_number = len(np.unique(y))
    if FLAGS.d_openml != None:
        '''
        lable convert from str to int in openml dataset
        '''
        y_0 = np.zeros((n_number, 1))
        for i in range(class_number):
            y_temp = np.where(y[:] != '%d' % i, -1, 1)
            y_0 = np.hstack((y_0, np.mat(y_temp).T))
        y= y_0[:, 1:]
        y = np.argmax(y, axis=1)
    else:
        y = np.array(np.where(y[:] != 1, -1, 1).reshape(n_number, 1))

    print('Training Linear SVM')
    d = 2 ** math.ceil(np.log2(f_num))
    x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S))
    clf = LinearSVC()
    clf.fit(x_value, y)
    print(clf.score(x_value, y))
    W_fcP = np.asmatrix(-np.ones((T * d, class_number)))
    n_number, f_num = np.shape(x)
    FLAGS.BATCHSIZE = n_number
    d = 2 ** math.ceil(np.log2(f_num))
    x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S))
    print(x_value.shape)
    x, y = x_test, y_test
    n_number, f_num = np.shape(x)
    if FLAGS.d_openml != None:
        '''
        lable convert from str to int in openml dataset
        '''
        y_0 = np.zeros((n_number, 1))
        for i in range(class_number):
            y_temp = np.where(y[:] != '%d' % i, -1, 1)
            y_0 = np.hstack((y_0, np.mat(y_temp).T))
        y= y_0[:, 1:]
        y = np.argmax(y, axis=1)
    else:
        y = np.array(np.where(y[:] != 1, -1, 1).reshape(n_number, 1))

    FLAGS.BATCHSIZE = n_number
    start = time.time()
    x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S))
    print(clf.score(x_value, y))
    print(time.time() - start, 'linear svm')


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
    name_space = ['CIFAR_10','Fashion-MNIST','mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    main(name)
