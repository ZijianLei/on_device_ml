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


def get_data(name):
    if FLAGS.d_openml != None:
        if name == 'CIFAR_10':
            x, y = sklearn.datasets.fetch_openml(name=name, return_X_y=True)
            x = x / 255
            x_train, x_test = x[:10000], x[10000:]
            y_train, y_test = y[:10000], y[10000:]

        else:
            x, y = sklearn.datasets.fetch_openml(name=name, return_X_y=True)
            x = x / 255
            x_train, x_test = x[:60000], x[60000:]
            y_train, y_test = y[:60000], y[60000:]
    else:
        if name == 'webspam' or 'covtype':
            X, y = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'train'))
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        else:
            x_train, y_train = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'train'))
            x_test, y_test = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'test'))
        x_train = x_train.todense()
        x_test = x_test.todense()
        print(x_train.shape)
    return x_train, y_train, x_test, y_test


# the fast implement of using hadamard transform to apporximate the gaussian random projection
def hadamard(d, f_num, batch, G, B, PI_value, S,T):
    x_ = batch
    n = x_.shape[0]
    x_ = np.pad(x_, ((0, 0), (0, d - f_num)), 'constant', constant_values=(0, 0))  # x.shape [batch,d]
    x_ = np.tile(x_, (1, T))
    x_i = np.multiply(x_, S)
    x_ = x_i.reshape(FLAGS.BATCHSIZE * T, d)
    for i in range(n * T):
        ffht.fht(x_[i])
    x_ = x_.reshape(FLAGS.BATCHSIZE, T * d)
    np.take(x_, PI_value, axis=1, mode='wrap', out=x_)
    x_transformed = np.multiply(x_, G)
    x_ = np.reshape(x_transformed, (FLAGS.BATCHSIZE * T, d))
    for i in range(n * T):
        ffht.fht(x_[i])
    x_ = x_.reshape(FLAGS.BATCHSIZE, T * d)
    x_value = np.multiply(x_, B)
    x_value = np.sign(x_value)
    # exit()
    return x_value


def optimization(x_value,y_temp,W,class_number):
    n_number, project_d = x_value.shape

    #original optimization method
    for c in range(class_number):
        W_temp = W[:,c]
        y_temp_c = y_temp[:,c]
        init= np.dot(x_value, W_temp )
        hinge_loss = sklearn.metrics.hinge_loss(y_temp_c, init)
        loss_new = np.sum(hinge_loss)+ sum(abs(W_temp))
        loss_old =  2*loss_new
        j  = 0
        while (loss_old-loss_new)/loss_old >= 1e-6:
            loss_old = loss_new
            for i in np.random.choice(project_d,project_d,replace=False):
                if W_temp[i] !=0:
                    derta = init-np.multiply(W_temp[i],x_value[:,i])
                    loss = sklearn.metrics.hinge_loss(y_temp_c,derta) + sum(abs(W_temp))*(10**(-j))
                    if loss < loss_new:
                        loss_new = loss
                        init = derta
                        W_temp[i] =0
                else:
                    w_i = np.sign(np.random.rand(1))
                    derta = init+np.multiply(w_i,x_value[:,i])
                    loss = sklearn.metrics.hinge_loss(y_temp_c,derta)+sum(abs(W_temp))*(10**(-j))
                    if loss < loss_new:
                        loss_new = loss
                        init = derta
                        W_temp[i] =w_i
                j +=1
        W[:,c] = W_temp
    return W
def label_processing(y,n_number):
    '''
            Start to processing the label
            '''
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
    return y_temp




def main(name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    '''
    read data and parameter initialize for the hadamard transform
    '''
    x_train, y_train, x_test, y_test = get_data(name)
    acc_linear = np.zeros(2)
    acc_binary = np.zeros(2)
    acc_random = np.zeros(2)
    time_linear = np.zeros(2)
    time_binary = np.zeros(2)
    time_random = np.zeros(2)
    marker = ['.','v','^','s','*','+']
    color = ['r','g','b','c','y','k']
    for iter in range(1):

        T = FLAGS.T

        x, y = x_train, y_train
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
        S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1) ** (-0.1)).reshape(T, -1))
        S = S.reshape(1, -1)
        FLAGS.BATCHSIZE = n_number
        class_number = len(np.unique(y))
        if class_number ==2:
            class_number -=1

        x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S,T))
        y_temp= label_processing(y,n_number)
        clf = LinearSVC()
        clf.fit(x_value, y)
        # print(clf.score(x_value,y))
        print('Training coordinate descent initiate from result of linear svm')

        n_number, project_d = x_value.shape
        test_number, f_num = np.shape(x_test)
        FLAGS.BATCHSIZE = test_number
        test_x = np.asmatrix(hadamard(d, f_num, x_test, G, B, PI_value, S,T))

        test_y = label_processing(y_test, test_number)
        # original optimization method
        range_lamda = [0,1e-4,5e-4,1e-3,5e-3,1e-2]
        for idx,lamda in enumerate(range_lamda):
            W_fcP = clf.coef_
            W_fcP = np.asmatrix(np.sign(W_fcP.reshape(-1, class_number)))
            W = W_fcP
            iter_loss = []
            iter_acc = []
            for c in range(class_number):
                W_temp = W[:, c]
                y_temp_c = y_temp[:, c]
                init = np.dot(x_value, W_temp)
                hinge_loss = sklearn.metrics.hinge_loss(y_temp_c, init) * n_number
                if class_number != 1:
                    predict = np.argmax(np.array(np.dot(test_x, W_temp)), axis=1)
                    y_lable = np.argmax(test_y, axis=1)
                    acc = accuracy_score(np.array(y_lable), np.array(predict))
                else:
                    predict = np.array(np.dot(test_x, W_temp))
                    acc = accuracy_score(np.sign(test_y), np.sign(predict))
                regularization = sum(abs(W_temp)) * lamda
                loss_new = np.sum(hinge_loss) + regularization
                iter_loss.append(loss_new)
                iter_acc.append(acc)
                loss_old = 2 * loss_new
                j = 0
                while (loss_old - loss_new) / loss_old >= 1e-6:
                    loss_old = loss_new
                    for i in np.random.choice(project_d, project_d, replace=False):
                        if W_temp[i] != 0:
                            w_i =W_temp[i]
                            w_i2= -W_temp[i]
                            derta = init - np.multiply(w_i, x_value[:, i])
                            regularization -= lamda
                            derta2 = init - 2 * np.multiply(w_i, x_value[:, i])
                            regularization_2 = regularization +  lamda
                            loss = sklearn.metrics.hinge_loss(y_temp_c, derta) + regularization
                            if loss < loss_new:
                                loss_new = loss
                                init = derta
                                W_temp[i] = 0
                            loss = sklearn.metrics.hinge_loss(y_temp_c, derta2) + regularization_2
                            if loss < loss_new:
                                loss_new = loss
                                init = derta2
                                W_temp[i] = w_i2
                                regularization = regularization_2
                        else:
                            derta = init + x_value[:, i]
                            regularization += lamda
                            derta2 = init - x_value[:, i]

                            loss = sklearn.metrics.hinge_loss(y_temp_c, derta) + regularization
                            if loss < loss_new:
                                loss_new = loss
                                init = derta
                                W_temp[i] = 1
                            loss = sklearn.metrics.hinge_loss(y_temp_c, derta2) + regularization
                            if loss < loss_new:
                                loss_new = loss
                                init = derta2
                                W_temp[i] = -1

                    if class_number != 1:
                        predict = np.argmax(np.array(np.dot(test_x, W_temp)), axis=1)
                        y_lable = np.argmax(test_y, axis=1)
                        acc = accuracy_score(np.array(y_lable), np.array(predict))
                    else:
                        predict = np.array(np.dot(test_x, W_temp))
                        acc = accuracy_score(np.sign(test_y), np.sign(predict))
                    iter_loss.append(loss_new)
                    iter_acc.append(acc)
                    j += 1
                W[:, c] = W_temp
                print(np.count_nonzero(W))
            clf2 = LinearSVC()
            clf2.fit(np.dot(x_value,W), y)
            acc = clf2.score(np.dot(test_x,W), test_y)
            # print(clf2.coef_)
            iter_acc.append(acc)
            #     plt.scatter(test_x,test_y)
            # plt.show()
            # exit()
            axis_x = np.arange(1,len(iter_acc)+1,1)
            print(iter_acc)
            ax1.plot(axis_x[:],iter_acc[:],linewidth=2,linestyle= '--',color =color[idx],label='%.1e_loss' % lamda,marker = marker[idx],
                     markersize=4)
                # ax2.plot(axis_x[:], iter_loss[:], linewidth = 2, linestyle = '--',color ='b', label = '%d_acc' % T,marker = marker[iter],
                #          markersize = 10)
    ax1.set_ylabel('acc')
    # ax2.set_ylabel('loss')
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    # box = ax2.get_position()
    # ax2.set_position([box.x0, box.y0 + box.height * 0.1,
    #                   box.width, box.height * 0.9])

    # Put a legend below current axis
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16),
    #            fancybox=True, shadow=True, ncol=5)
    plt.savefig('%s_%s.png' % (name, 'lamda_acc'))
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
