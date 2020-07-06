from memory_profiler import profile
import  warnings
import ffht
import math
import time
from sklearn.metrics import *
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
    move = np.max(x_train)+np.min(x_train)
    x_train = 2*x_train-move
    x_test = 2*x_test-move
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

def fastfood(d,f_num,batch,G,B,PI_value,S,FLAGS,sigma = 1):
    p = FLAGS.p
    x_ = batch
    n = x_.shape[0]
    x_ = np.pad(x_, ((0, 0), (0, d - x_.shape[1])), 'constant', constant_values=(0, 0))
    x_ = np.tile(x_, (1, p))
    x_i = np.multiply(x_, S)
    x_ = x_i.reshape(FLAGS.data_number * p, d)
    for i in range(n*p):
        ffht.fht(x_[i])
    x_ = x_.reshape(FLAGS.data_number, p * d)
    x_transformed = np.multiply(x_, G)
    np.take(x_, PI_value, axis=1, mode='wrap', out=x_)
    x_ = np.reshape(x_transformed, (FLAGS.data_number * p, d))
    for i in range(n*p):
        ffht.fht(x_[i])
    x_ = x_.reshape(FLAGS.data_number, p * d)
    x_value = np.multiply(x_, B)/(p*d)
    x_value = np.sqrt(2*sigma) * x_value / np.sqrt(d*p)
    x_value = np.cos(x_value)
    return x_value

def feature_binarizing(d,f_num,batch,G,B,PI_value,S,FLAGS,sigma = 1):
    '''
    :param d:  origin dimension of data d = 2**q
    :param batch:  the original data x
    :param G, B, PI_value, S: Paramters of FastFood Kernel approximation
    :param sigma: rescaling the transformed data
    :return:  return the binary kernel approximation of data sign(cos(RX+b)+t)
    '''
    p = FLAGS.p
    x_ = batch
    n = x_.shape[0]
    x_ = np.pad(x_, ((0, 0), (0, d - x_.shape[1])), 'constant', constant_values=(0, 0))
    x_ = np.tile(x_, (1, p))
    x_i = np.multiply(x_, S)
    x_ = x_i.reshape(n * p, d)
    for i in range(n*p):
        ffht.fht(x_[i])
    x_ = x_.reshape(n, p * d)
    x_transformed = np.multiply(x_, G)
    np.take(x_, PI_value, axis=1, mode='wrap', out=x_)
    x_ = np.reshape(x_transformed, (n * p, d))
    for i in range(n*p):
        ffht.fht(x_[i])
    x_ = x_.reshape(n, p * d)
    x_value = np.multiply(x_, B)
    x_value = x_value/(sigma*(np.sqrt(p*d)**3))
    x_value = np.cos(x_value+FLAGS.b)+FLAGS.t
    x_value = np.sign(x_value)
    return x_value


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
            y_temp = np.array(np.where(y[:] != 1, 0, 1).reshape(n_number, 1))
            class_number -= 1
        else:
            y_0 = np.zeros((n_number, 1))
            y = y - 1
            for i in range(class_number):
                y_temp = np.where(y[:] != i, 0, 1).reshape(n_number, 1)
                y_0 = np.hstack((y_0, y_temp))
            y_temp = y_0[:, 1:]
    return y_temp


def optimization(x_value,y_temp,W,class_number,lamda = 0):
    n_number, project_d = x_value.shape
    x_value = np.asmatrix(x_value)
    lamda = lamda/project_d
    for c in range(class_number):
        W_temp = np.asmatrix(W[:,c])
        y_temp_c = np.asmatrix(y_temp[:,c]).reshape(-1,1)
        temp_store = x_value.dot(W_temp)
        init= 1-np.multiply(y_temp_c,temp_store)
        hinge_loss = np.sum(np.clip(init,0,None))/n_number
        regularization = np.count_nonzero(W_temp)*lamda
        loss_new = np.sum(hinge_loss)+ regularization
        loss_old = 2*loss_new
        while (loss_old-loss_new)/loss_old >= 1e-6:
            loss_old = loss_new
            for i in range(project_d):
                if W_temp[i] != 0:
                    w_i = W_temp[i]
                    w_i2 = -W_temp[i]
                    temp = np.multiply(y_temp_c,x_value[:,i]*w_i)
                    derta = init + temp
                    regularization -= lamda
                    derta2 = init + 2 * temp
                    regularization_2 = regularization + lamda
                    loss = np.sum(np.clip(derta,0,None))/n_number + regularization
                    if loss < loss_new:
                        loss_new = loss
                        init = derta
                        W_temp[i] = 0
                    loss = np.sum(np.clip(derta2,0,None))/n_number + regularization_2
                    if loss < loss_new:
                        loss_new = loss
                        init = derta2
                        W_temp[i] = w_i2
                        regularization = regularization_2

                else:
                    temp = np.multiply(y_temp_c,x_value[:, i])
                    derta = init - temp  # change to 1
                    regularization += lamda
                    derta2 = init + temp  # change to -1

                    loss = np.sum(np.clip(derta, 0, None)) / n_number + regularization
                    if loss < loss_new:
                        loss_new = loss
                        init = derta
                        W_temp[i] = 1
                    loss = np.sum(np.clip(derta2,0,None))/n_number  + regularization
                    if loss < loss_new:
                        loss_new = loss
                        init = derta2
                        W_temp[i] = -1
        W[:,c] = W_temp
    return W
def optimization2(init_W,class_number):
    print('Training coordinate descent initiate from result of linear svm using cos similarity')
    #original optimization method
    W = np.empty((len(init_W[:,0]),class_number))
    for c in range(class_number):
        W_temp = np.sign(init_W[:,c])
        length = len(W_temp)
        cos_distance = cosine(W_temp,init_W[:,c])
        loss_new = cos_distance
        loss_old =  2*loss_new
        j = 0
        while loss_old-loss_new  != 0:
            loss_old = loss_new
            for i in np.random.choice(length, length, replace=False):
                if W_temp[i] != 0:
                    a = copy.deepcopy(W_temp[i])
                    W_temp[i] = 0
                    loss1 = cosine(W_temp,init_W[:,c])
                    W_temp[i] = -a
                    loss2 = cosine(W_temp, init_W[:,c])

                    if loss1 < loss_new:
                        loss_new = loss1
                        W_temp[i] = 0
                    elif loss2<loss_new:
                        loss_new = loss2
                    else:
                        W_temp[i] = a
                else:
                    W_temp[i] = 1
                    loss1 = cosine(W_temp,init_W[:,c])
                    W_temp[i] = -1
                    loss2 = cosine(W_temp, init_W[:,c])
                    if loss1 < loss_new:
                        loss_new = loss1
                        W_temp[i] = 1
                    elif loss2<loss_new:
                        loss_new = loss2
                        W_temp[i] = -1
                    else:
                        W_temp[i] = 0
        W[:,c] = W_temp
    return W

def predict_acc(x_value,y_temp,test_x,test_y,W):
    clf2 = LinearSVC(dual = False)
    clf2.fit(np.dot(x_value, W), y_temp)
    # print(clf2.score(np.dot(x_value, W), y_temp))
    acc = clf2.score(np.dot(test_x, W), test_y)
    return  acc