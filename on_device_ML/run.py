from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.linalg import hadamard
from sklearn_extra.utils._cyfht import fht as cyfht2
import timeit
from scipy.linalg import hadamard
import argparse

import ffht
import math
import time
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file
from sklearn_extra.kernel_approximation import Fastfood
from sklearn.svm import  SVC,LinearSVC
import scipy
import numpy as np
from scipy.stats import chi
import sklearn
from scipy._lib._util import _asarray_validated
from sklearn.utils import check_array, check_random_state

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

# @profile
# the fast implement of using hadamard transform to apporximate the gaussian random projection
def hadamard(d,f_num,batch,G,B,PI_value,S):
    T = FLAGS.T
    x_ = batch
    n = x_.shape[0]
    x_ = np.pad(x_, ((0,0),(0, d - f_num)), 'constant', constant_values=(0, 0))  # x.shape [batch,d]
    x_ = np.tile(x_, (1, T))
    x_i = np.multiply(x_, S)
    x_ = x_i.reshape(FLAGS.BATCHSIZE*T, d)
    for i in range(x_.shape[0]):
        ffht.fht(x_[i])
    x_ =  x_.reshape(FLAGS.BATCHSIZE, d * T)
    x_transformed = np.multiply(x_, G)
    x_ = np.reshape(x_transformed, (FLAGS.BATCHSIZE*T, d))
    for i in range(x_.shape[0]):
        ffht.fht(x_[i])
        # x_[i] = np.sign(x_[i])
    x_ = x_.reshape(FLAGS.BATCHSIZE, T * d)
    x_ = np.sign(x_)
    x_value = np.multiply(x_, B)

    # #print(x_value)
    return x_value


def optimization(x_value,y_temp,W,class_number):
    n_number, project_d = x_value.shape
    for c in range(class_number):
        W_temp = W[:,c]
        y_temp_c = y_temp[:,c]
        init= np.dot(x_value, W_temp )
        hinge_loss = sklearn.metrics.hinge_loss(y_temp_c, init)*n_number
        loss_init = np.sum(hinge_loss)
        loss_0 = 0
        while loss_0 != loss_init:
            loss_0 = loss_init
            for i in range(project_d):
                derta = init-np.multiply(W_temp[i],x_value[:,i])*2
                loss = sklearn.metrics.hinge_loss(y_temp_c,derta)*n_number
                if loss < loss_init:
                    loss_init = loss
                    init = derta
                    W_temp[i] = -W_temp[i]
        W[:,c] = W_temp
    return W

def main(name ):
    '''
    read data and parameter initialize for the hadamard transform
    '''
    x_train, y_train, x_test, y_test= get_data(name)
    time_hadamard_my= np.zeros(3)
    time_extra = np.zeros(3)
    time_random = np.zeros(3)

    x,y = x_train,y_train
    for iter in range(3):
        n_number, f_num = np.shape(x)
        d = 2 ** math.ceil(np.log2(f_num))
        T = FLAGS.T
        # rng =
        G = np.random.randn(T * d)
        B = np.random.uniform(-1, 1, T * d)
        B = np.sign(B)
        PI_value = np.random.permutation(d)
        G_fro = G.reshape(T, d)
        s_i = chi.rvs(d, size=(T, d))
        S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1) ** (-0.5)).reshape(T, -1))
        S = S.reshape(1, -1)
        FLAGS.BATCHSIZE = n_number

        ff_transform = Fastfood(n_components=d*T,tradeoff_mem_accuracy="mem")
        #test the time for my method
        start = time.time()
        x_value = np.asmatrix(hadamard(d, f_num, x, G, B, PI_value, S))
        time_hadamard_my[iter] = time.time()-start
        #test the time for fastfood from sklean-extra

        pars = ff_transform.fit(x)
        start = time.time()
        x_value = np.asmatrix(np.sign(pars.transform(x)))
        time_extra[iter] = time.time() - start
        start = time.time()
        hash_plan = np.random.randn(f_num, T * d)
        x_value = np.sign(np.dot(x,hash_plan))
        time_random[iter] = time.time() - start
    print(np.mean(time_hadamard_my),np.mean(time_extra),np.mean(time_random))
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
    main(name)