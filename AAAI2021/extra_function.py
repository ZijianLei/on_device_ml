from memory_profiler import profile
import  warnings
import ffht
import math
import time
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from sklearn.svm import  LinearSVC,SVC
from sklearn.model_selection import  train_test_split
import scipy
from sklearn.preprocessing import *
import numpy as np
from scipy.stats import chi
import sklearn
import copy
from scipy.spatial.distance import cosine
from sklearn.model_selection import KFold
from scipy._lib._util import _asarray_validated
import gc
from sklearn.utils import check_array, check_random_state
def get_data(name,FLAGS):
    '''
    Read the data from openml or Libsvm dataset.
    '''
    if FLAGS.d_openml != None:
        x,y= sklearn.datasets.fetch_openml(name = name,return_X_y= True)
        x = x / 255
        x_train,x_test = x[:60000],x[60000:]
        y_train,y_test = y[:60000],y[60000:]
    else:
        if name == 'webspam' or name == 'covtype':
            X,y = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" %(name,'train'))
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        elif name == 'usps':
            x_train, y_train = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'usps'))
            x_test, y_test = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'usps.t'))
        else:
            x_train,y_train = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" %(name,'train'))
            x_test,y_test = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'test'))
        x_train = x_train.todense()
        x_test = x_test.todense()
    # move = np.max(x_train)+np.min(x_train)
    # x_train = 2*x_train-move
    # x_test = 2*x_test-move
    scaler = preprocessing.StandardScaler(with_std=False).fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print(np.max(x_train), np.min(x_train), np.mean(x_train))
    return x_train,y_train,x_test,y_test


def parameters(n_number,p,d,FLAGS):
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
    FLAGS.b = np.random.uniform(0, 2 * math.pi, d * p)
    FLAGS.t = np.random.uniform(-1 , 1 , d * p)
    return PI_value,G,B,S

def label_processing(y,n_number,FLAGS):
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
            y_temp = np.where(y[:] != '%d' % i, 0, 1)
            y_0 = np.hstack((y_0, np.mat(y_temp).T))
        y_temp = y_0[:, 1:]
    else:
        '''
        for date from libsvm dataset both binary classification and multi-classification problem
        '''
        if class_number == 2:
            # y_temp = np.array(np.where(y[:] != 1, 0, 1).reshape(n_number, 1))
            y_0 = np.zeros((n_number, 1))
            if FLAGS.d_libsvm == 'covtype':
                y = y - 1
            if FLAGS.d_libsvm == 'webspam':
                y[y == -1] = 0
            for i in range(class_number):
                y_temp = np.where(y[:] != i, 0, 1).reshape(n_number, 1)
                y_0 = np.hstack((y_0, y_temp))
            y_temp = y_0[:, 1:]
            # class_number -= 1
        else:
            y_0 = np.zeros((n_number, 1))
            y = y - 1
            for i in range(class_number):
                y_temp = np.where(y[:] != i, 0, 1).reshape(n_number, 1)
                y_0 = np.hstack((y_0, y_temp))
            y_temp = y_0[:, 1:]
    return y_temp
def label_processing_hinge(y,n_number,FLAGS):
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
            # y_temp = np.array(np.where(y[:] != 1, 0, 1).reshape(n_number, 1))
            y_0 = np.zeros((n_number, 1))
            if FLAGS.d_libsvm == 'covtype':
                y = y - 1
            if FLAGS.d_libsvm == 'webspam':
                y[y == -1] = -1
            for i in range(class_number):
                y_temp = np.where(y[:] != i, -1, 1).reshape(n_number, 1)
                y_0 = np.hstack((y_0, y_temp))
            y_temp = y_0[:, 1:]
            # class_number -= 1
        else:
            y_0 = np.zeros((n_number, 1))
            y = y - 1
            for i in range(class_number):
                y_temp = np.where(y[:] != i, -1, 1).reshape(n_number, 1)
                y_0 = np.hstack((y_0, y_temp))
            y_temp = y_0[:, 1:]
    return y_temp

