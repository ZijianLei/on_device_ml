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
def main(name):
    x_train, y_train, x_test, y_test = get_data(name)
    x, y = x_train, y_train
    n_number, f_num = np.shape(x)
    x_test = np.pad(x_test, ((0,0),(0, f_num-x_test.shape[1] )), 'constant', constant_values=(0, 0))  # x.shape [batch,d]
    class_number =len(np.unique(y))
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

    # clf = SVC(gamma='auto')
    clf = LinearSVC()
    print('train')

    clf.fit(x,y)
    print(clf.score(x, y))
    x, y = x_test,y_test
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
    start = time.time()
    print(clf.score(x,y))
    print(time.time() - start,'linear svm')

    x, y = x_train, y_train
    n_number, f_num = np.shape(x)
    class_number = len(np.unique(y))
    if FLAGS.d_openml != None:
        '''
        lable convert from str to int in openml dataset
        '''
        y_0 = np.zeros((n_number, 1))
        for i in range(class_number):
            y_temp = np.where(y[:] != '%d' % i, -1, 1)
            y_0 = np.hstack((y_0, np.mat(y_temp).T))
        y = y_0[:, 1:]
        y = np.argmax(y, axis=1)
    else:
        if class_number == 1:
            y= np.array(np.where(y[:] != 1, -1, 1).reshape(n_number, 1))
        else:
            y = y - 1


    clf = SVC(gamma='auto')
    # clf = LinearSVC()
    print('train')

    clf.fit(x, y)
    print(clf.score(x, y))
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
        y = y_0[:, 1:]
        y = np.argmax(y, axis=1)
    else:
        if class_number == 1:
            y = np.array(np.where(y[:] != 1, -1, 1).reshape(n_number, 1))
        else:

            y = y - 1

    start = time.time()
    print(clf.score(x, y))
    print(time.time() - start,'kernel svm')
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
