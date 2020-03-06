from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#import scikit_image-0.12.3.dist-info
import argparse
import sys
import math
import sklearn
import time
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file
from sklearn.metrics.pairwise import  *
import scipy
import numpy as np
from sklearn.svm import LinearSVC
from scipy.stats import chi


def lsh1(data,hash_pool,output_bit,n_number,temp_plan):
    lsh_map = np.matrix(np.zeros((n_number,output_bit)))
    for i in range(output_bit):
        lsh_map[:,i] = np.sign(np.dot(data[:,:],hash_pool[temp_plan[i]]))
    return lsh_map

def lsh2(data,hash_plan):
    lsh_map = np.sign(np.dot(data,hash_plan))
    return lsh_map


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

def main(name,set_lsh_function):# 0 for naive random projection and 1 for pool trick

    x_train, y_train, x_test, y_test = get_data(name)
    x, y = x_train, y_train
    # x = x.todense()
    class_number = len(np.unique(y))
    n_number, f_num = np.shape(x)
    x_test = np.pad(x_test, ((0, 0), (0, f_num - x_test.shape[1])), 'constant', constant_values=(0, 0))
    clf = LinearSVC()
    d = 2 ** math.ceil(np.log2(f_num))
    T = FLAGS.T
    pool_size = int(T*d/5)
    if set_lsh_function == 1:
        hash_pool = np.random.randn(pool_size,1)
        temp_plan = np.random.random_integers(0,pool_size-1,size=(T*d,d))
        x_value = lsh1(x, hash_pool, T * d, n_number, temp_plan)
    else:
        hash_plan = np.random.randn(f_num, T * d)
        x_value = lsh2(x, hash_plan)
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
    clf.fit(x_value, y)
    print('train LSH with Liblinear', clf.score(x_value, y))
    print('Training LSH')
    W_fcP = clf.coef_

    W_fcP = np.asmatrix(np.sign(W_fcP.reshape(-1, class_number)))
    for c in range(class_number):
        W_temp = W_fcP[:,c]
        y_temp_c = y_temp[:,c]
        init= np.dot(x_value, W_temp )
        hinge_loss = sklearn.metrics.hinge_loss(y_temp_c, init)*n_number
        loss_new = np.sum(hinge_loss)
        loss_old =  2*loss_new
        while (loss_old-loss_new)/loss_old >= 1e-5:
            loss_old = loss_new
            for i in range(x_value.shape[1]):
                derta = init-np.multiply(W_temp[i],x_value[:,i])*2
                loss = sklearn.metrics.hinge_loss(y_temp_c,derta)*n_number
                if loss < loss_new:
                    loss_new = loss
                    init = derta
                    W_temp[i] = -W_temp[i]
        W_fcP[:,c] = W_temp
    if class_number != 1:
        predict = np.argmax(np.array(np.dot(x_value, W_fcP)), axis=1)
        y_lable = np.argmax(y_temp, axis=1)
        # print(y[:10],y_lable[:10],predict[:10])
        acc = accuracy_score(np.array(y_lable), np.array(predict))
        print(acc,'train')
    else:
        predict = np.array(np.dot(x_value, W_fcP))
        y_lable = y_temp
        acc = accuracy_score(np.sign(y_lable), np.sign(predict))
        print(acc, 'train')




    x, y = x_test, y_test
    n_number, f_num = np.shape(x)
    FLAGS.BATCHSIZE = n_number
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
    # x = x.todense()
    # x = sklearn.preprocessing.scale(x)
    start = time.time()
    if set_lsh_function == 1:
        x_value = lsh1(x, hash_pool, T * d, n_number, temp_plan)
    else:
        x_value = lsh2(x, hash_plan)
    time1 = time.time() - start

    if class_number != 1:
        y_lable = np.argmax(y_temp, axis=1)
        start = time.time()
        predict = np.argmax(np.array(np.dot(x_value, W_fcP)), axis=1)
        acc = accuracy_score(np.array(y_lable), np.array(predict))
        print(acc, 'test')
    else:
        y_lable = y_temp
        start = time.time()
        predict = np.array(np.dot(x_value, W_fcP))
        acc = accuracy_score(np.sign(y_temp), np.sign(predict))
        print(acc, 'test')
    print(time1+time.time() - start,'test time for LSH coordinate descent')
    start = time.time()
    print('LSH+liblinear predict',clf.score(x_value,y_lable))
    print(time1+time.time() - start)


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

    '''
            available data:
            0:CIFAR_10,
            1:Fashion-MNIST,
            2:mnist_784'
        '''
    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name_space = ['CIFAR_10', 'Fashion-MNIST', 'mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    main(name,0)