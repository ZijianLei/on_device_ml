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


def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
  a = _asarray_validated(a, check_finite=False)
  if b is not None:
    a, b = np.broadcast_arrays(a, b)
    if np.any(b == 0):
      a = a + 0.  # promote to at least float
      a[b == 0] = -np.inf

  a_max = np.amax(a, axis=axis, keepdims=True)

  if a_max.ndim > 0:
    a_max[~np.isfinite(a_max)] = 0
  elif not np.isfinite(a_max):
    a_max = 0

  if b is not None:
    b = np.asarray(b)
    tmp = b * np.exp(a - a_max)
  else:
    tmp = np.exp(a - a_max)

  # suppress warnings about log of zero
  with np.errstate(divide='ignore'):
    s = np.sum(tmp, axis=axis, keepdims=keepdims)
    if return_sign:
      sgn = np.sign(s)
      s *= sgn  # /= makes more sense but we need zero -> zero
    out = np.log(s)
  if not keepdims:
    a_max = np.squeeze(a_max, axis=axis)
  out += a_max

  if return_sign:
    return out, sgn
  else:
    return out

def softmax(x):
    return np.exp(x - logsumexp(x, axis=1, keepdims=True))

# the fast implement of using hadamard transform to apporximate the gaussian random projection
def hadamard(d,f_num,batch,G,B,PI_value,S):
    T = FLAGS.T
    x_ = batch
    x_ = np.pad(x_, ((0,0),(0, d - f_num)), 'constant', constant_values=(0, 0))  # x.shape [batch,d]
    x_ = np.tile(x_, (1, T))
    x_i = np.multiply(x_, S)
    x_ = x_i.reshape(FLAGS.BATCHSIZE, T, d)
    h = 1
    for i in range(x_.shape[0]):
        for j in range(T):
            ffht.fht(x_[i,j])
    x_transformed = np.multiply(x_.reshape(FLAGS.BATCHSIZE, d * T), G)
    x_transformed = np.reshape(x_transformed, (FLAGS.BATCHSIZE, T, d))
    # x_permutation = x_transformed[:, :, PI_value]
    # x_ = x_permutation
    x_ = x_transformed
    h = 1
    for i in range(x_.shape[0]):
        for j in range(T):
            ffht.fht(x_[i, j])
    # while h < d:
    #     for i in range(0, d, h * 2):
    #         for j in range(i, i + h):
    #             a = x_[:, :, j]
    #             b = x_[:, :, j + h]
    #             temp = a - b
    #             x_[:, :, j] = a + b
    #             x_[:, :, j + h] = temp
    #     h *= 2

    x_ = x_.reshape(FLAGS.BATCHSIZE, T * d)
    x_value = np.multiply(x_, B)
    x_value = np.sign(x_value)
    # print(x_value)
    return x_value


def main(name ):
    #read the dataset and initialize the parameter for hadamard transform
    x_train, y_train, x_test, y_test= get_data(name)
    x,y = x_train,y_train
    n_number, f_num = np.shape(x)
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
        y_temp = y_0[:, 1:]
    else:
        '''
        for date from libsvm dataset both binary classification and multi-classification problem
        '''
        if class_number == 2:
            y_temp = np.array(np.where(y[:] != 1, -1, 1).reshape(n_number, 1))
            class_number -= 1
        else:
            y_0 = np.zeros((n_number, 1))
            y = y - 1
            for i in range(class_number):
                y_temp = np.where(y[:] != i, -1, 1).reshape(n_number, 1)
                y_0 = np.hstack((y_0, y_temp))
            y_temp = y_0[:, 1:]

    print('Training coordinate descent')
    W_fcP = np.asmatrix(-np.ones((T * d, class_number)))
    n_number, f_num = np.shape(x)
    FLAGS.BATCHSIZE = n_number
    d = 2 ** math.ceil(np.log2(f_num))
    x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S))
    print(x_value.shape)
    for c in range(class_number):
        W_temp = np.asmatrix(-np.ones((T * d,1)))
        y_temp_c = y_temp[:,c]
        init= np.dot(x_value, W_temp )
        #loss_init = sklearn.metrics.mean_squared_error(y, predict)
        # hinge_loss = np.max(np.hstack((np.zeros((n_number,1)),-np.multiply(y_temp_c,predict))),axis = 1)
        hinge_loss = sklearn.metrics.hinge_loss(y_temp_c, init)*n_number
        loss_init = np.sum(hinge_loss)
        loss_0 = 0
        j = 1
        print(loss_init)

        while loss_0 != loss_init:
            loss_0 = loss_init
            # predict = np.multiply(W_temp.T, x_value)*2
            for i in range(T*d):
                derta = init-np.multiply(W_temp[i],x_value[:,i])*2
                # derta = init - predict[:,i]
                # hinge_loss = np.max(np.hstack((np.zeros((n_number,1)),-np.multiply(y_temp_c,derta))),axis = 1)
                loss = sklearn.metrics.hinge_loss(y_temp_c,derta)*n_number
                if loss < loss_init:
                    loss_init = loss
                    init = derta
                    W_temp[i] = -W_temp[i]
            # print(loss_init,j)
            j+=1
        W_fcP[:,c] = W_temp

    if class_number != 1:
        predict = np.argmax(np.array(np.dot(x_value, W_fcP)), axis=1)
        y_lable = np.argmax(y_temp, axis=1)
        acc = accuracy_score(np.array(y_lable), np.array(predict))
        print(acc,'train')
    else:
        predict = np.array(np.dot(x_value, W_fcP))
        acc = accuracy_score(np.sign(y_temp), np.sign(predict))
        print(acc, 'train')

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
        y_temp = y_0[:, 1:]
    else:
        if class_number == 1:
            y_temp = np.array(np.where(y[:] != 1, -1, 1).reshape(n_number, 1))
        else:
            y_0 = np.zeros((n_number, 1))
            y = y - 1
            for i in range(class_number):
                y_temp = np.where(y[:] != i, -1, 1).reshape(n_number, 1)
                y_0 = np.hstack((y_0, y_temp))
            y_temp = y_0[:, 1:]


    n_number, f_num = np.shape(x)
    FLAGS.BATCHSIZE = n_number
    start = time.time()
    x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S))
    if class_number != 1:
        predict = np.argmax(np.array(np.dot(x_value, W_fcP)), axis=1)
        y_lable = np.argmax(y_temp, axis=1)
        acc = accuracy_score(np.array(y_lable), np.array(predict))
        print(acc, 'test')
    else:
        predict = np.array(np.dot(x_value, W_fcP))
        acc = accuracy_score(np.sign(y_temp), np.sign(predict))
        print(acc, 'test')
    print(time.time() - start)

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
