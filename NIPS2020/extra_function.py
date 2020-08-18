from memory_profiler import profile
import  warnings
import ffht
import math
import time
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file
from sklearn_extra.kernel_approximation import Fastfood
from sklearn.svm import  SVC,LinearSVC
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
    if FLAGS.d_openml != None:
        if name == 'CIFAR_10':
            x, y = sklearn.datasets.fetch_openml(name=name, return_X_y=True)
            x = x/255
            x_train, x_test = x[:10000], x[10000:]
            y_train, y_test = y[:10000], y[10000:]

        else:
            x,y= sklearn.datasets.fetch_openml(name = name,return_X_y= True)
            x = x / 255
            x_train,x_test = x[:60000],x[60000:]
            y_train,y_test = y[:60000],y[60000:]
    else:
        if name == 'webspam' or name == 'covtype':
            print(0)
            X,y = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" %(name,'train'))
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        elif name == 'usps':
            print(1)
            x_train, y_train = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'usps'))

            x_test, y_test = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'usps.t'))
        else:
            x_train,y_train = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" %(name,'train'))
            x_test,y_test = load_svmlight_file("../svm/BudgetedSVM/original/%s/%s" % (name, 'test'))
        x_train = x_train.todense()
        x_test = x_test.todense()
        # scaler =StandardScaler().fit(x_train)
        # f_num = x_train.shape[1]
        # x_test = np.pad(x_test, ((0, 0), (0, f_num - x_test.shape[1])), 'constant', constant_values=(0, 0))
        # x_train =scaler.transform(x_train)
        # x_test = scaler.transform(x_test)
    move = np.max(x_train)+np.min(x_train)
    x_train = 2*x_train-move
    x_test = 2*x_test-move
    return x_train,y_train,x_test,y_test

def fastfood_value(n_number,T,d,FLAGS,method = 1 ):
    FLAGS.T = T
    G = np.random.randn(T * d)
    B = np.random.uniform(-1, 1, T * d)
    B[B > 0] = 1
    B[B < 0] = -1
    # PI_value = np.hstack
    PI_value = np.hstack([(i * d) + np.random.permutation(d) for i in range(T)])
    G_fro = G.reshape(T, d)
    s_i = chi.rvs(d, size=(T, d))
    S = np.multiply(s_i, np.array(np.linalg.norm(G_fro, axis=1)).reshape(T, -1))
    S = S.reshape(1, -1)
    FLAGS.BATCHSIZE = n_number
    FLAGS.b = None
    FLAGS.t = None
    if method == 1:
        FLAGS.b = np.random.uniform(0, 2 * math.pi, d * T)
        FLAGS.t = np.random.uniform(-1, 1, d * T)
    return PI_value,G,B,S

def fastfood(d,f_num,batch,G,B,PI_value,S,FLAGS,sigma = 1):
    start = time.time()
    T = FLAGS.T

    x_ = batch
    n = x_.shape[0]
      # x.shape [batch,d]
    x_ = np.pad(x_, ((0, 0), (0, d - x_.shape[1])), 'constant', constant_values=(0, 0))
    x_ = np.tile(x_, (1, T))
    x_i = np.multiply(x_, S)
    x_ = x_i.reshape(FLAGS.BATCHSIZE * T, d)
    for i in range(n*T):
        ffht.fht(x_[i])
    x_ = x_.reshape(FLAGS.BATCHSIZE, T * d)
    x_transformed = np.multiply(x_, G)
    np.take(x_, PI_value, axis=1, mode='wrap', out=x_)

    x_ = np.reshape(x_transformed, (FLAGS.BATCHSIZE * T, d))
    for i in range(n*T):
        ffht.fht(x_[i])
    x_ = x_.reshape(FLAGS.BATCHSIZE, T * d)
    # print(np.linalg.norm(x_[:2],axis=1))
    # print(np.mean(x_[:,:2], axis=0))
    x_value = np.multiply(x_, B)
    # print(np.linalg.norm(x_value[:2], axis=1))

    # print(np.any(FLAGS.b))
    x_value = (sigma**2) * x_value / np.sqrt(d)
    x_value = np.cos(x_value)
    # print(time.time()-start)
    return x_value

# the fast implement of using hadamard transform to apporximate the gaussian random projection
def hadamard(d,f_num,batch,G,B,PI_value,S,FLAGS,sigma = 1):
    start = time.time()
    T = FLAGS.T
    kf = KFold(n_splits=2)
    x_return = np.zeros((1,T*d))
    for train_index,test_index in kf.split(batch):
        x_ = batch[test_index]
        # print(x_.shape)
        n = x_.shape[0]
          # x.shape [batch,d]
        x_ = np.pad(x_, ((0, 0), (0, d - x_.shape[1])), 'constant', constant_values=(0, 0))
        x_ = np.tile(x_, (1, T))
        x_i = np.multiply(x_, S)
        x_ = x_i.reshape(n * T, d)
        for i in range(n*T):
            ffht.fht(x_[i])
        x_ = x_.reshape(n, T * d)
        x_transformed = np.multiply(x_, G)
        np.take(x_, PI_value, axis=1, mode='wrap', out=x_)

        x_ = np.reshape(x_transformed, (n * T, d))
        for i in range(n*T):
            ffht.fht(x_[i])
        x_ = x_.reshape(n, T * d)
        # print(np.linalg.norm(x_[:2],axis=1))
        # print(np.mean(x_[:,:2], axis=0))
        x_value = np.multiply(x_, B)
        # print(np.linalg.norm(x_value[:2], axis=1))

        # print(np.any(FLAGS.b))
        if np.any(FLAGS.b) :
            # print(1)
            x_value = x_value / (sigma *np.sqrt(d)**3)
            x_value = np.cos(x_value+FLAGS.b)+FLAGS.t
        # print(time.time()-start)
        x_value = np.sign(x_value)
        x_return = np.vstack((x_return,x_value))

    return x_return[1:,:]

def hadamard2(d,f_num,batch,G,B,PI_value,S,FLAGS,sigma = 1):
    p = FLAGS.p
    x_ = batch
    n = x_.shape[0]
    x_ = np.pad(x_, ((0, 0), (0, d - x_.shape[1])), 'constant', constant_values=(0, 0))
    x_ = np.tile(x_, (1, p))

    x_i = np.multiply(x_, B)
    x_ = x_i.reshape(n * p, d)
    for i in range(n * p):
        ffht.fht(x_[i])
    x_ = x_.reshape(n, p * d)
    np.take(x_, PI_value, axis=1, mode='wrap', out=x_)
    x_transformed = np.multiply(x_, G)

    x_ = np.reshape(x_transformed, (n * p, d))
    for i in range(n * p):
        ffht.fht(x_[i])
    x_ = x_.reshape(n, p * d)
    x_value = np.multiply(x_, S)

    x_value = x_value / (sigma * (np.sqrt(d) ** 3))
    x_value = np.cos(x_value + FLAGS.b) + FLAGS.t
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
    # print(y_temp.shape)
    return y_temp

def optimization2(init_W,class_number):
    print('Training coordinate descent initiate from result of linear svm using cos similarity')
    #original optimization method
    W = np.empty((class_number,len(init_W[0])))
    for c in range(class_number):
        W_temp = np.sign(init_W[c])
        length = len(W_temp)
        cos_distance = cosine(W_temp,init_W[c])
        loss_new = cos_distance
        loss_old =  2*loss_new
        j = 0
        while loss_old-loss_new  != 0:
            loss_old = loss_new
            for i in np.random.choice(length, length, replace=False):
                if W_temp[i] != 0:
                    a = copy.deepcopy(W_temp[i])
                    W_temp[i] = 0
                    loss1 = cosine(W_temp,init_W[c])
                    W_temp[i] = -a
                    loss2 = cosine(W_temp, init_W[c])

                    if loss1 < loss_new:
                        loss_new = loss1
                        W_temp[i] = 0
                    elif loss2<loss_new:
                        loss_new = loss2
                    else:
                        W_temp[i] = a
                else:
                    W_temp[i] = 1
                    loss1 = cosine(W_temp,init_W[c])
                    W_temp[i] = -1
                    loss2 = cosine(W_temp, init_W[c])
                    if loss1 < loss_new:
                        loss_new = loss1
                        W_temp[i] = 1
                    elif loss2<loss_new:
                        loss_new = loss2
                        W_temp[i] = -1
                    else:
                        W_temp[i] = 0
        W[c,:] = W_temp
    return W.T
def optimization(x_value,y_temp,W,class_number,lamda = 0):
    print('Training coordinate descent initiate from result of linear svm')
    n_number, project_d = x_value.shape
    #original optimization method
    x_value = np.asmatrix(x_value)
    lamda = lamda/project_d
    for c in range(class_number):
        # print(c)
        W_temp = np.asmatrix(W[:,c])
        y_temp_c = np.asmatrix(y_temp[:,c]).reshape(-1,1)
        # print(y_temp_c.shape,x_value.shape,W_temp.shape)
        temp_store = x_value.dot(W_temp)
        # print(temp_store.shape,y_temp_c.shape)
        init= -np.multiply(y_temp_c,temp_store)
        hinge_loss = np.sum(np.clip(init,0,None))/n_number
        regularization = np.count_nonzero(W_temp)*lamda
        loss_new = np.sum(hinge_loss)+ regularization
        loss_old =  2*loss_new
        j = 0
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
                    derta = init - temp # change to 1
                    regularization += lamda
                    derta2 = init + temp # change to -1

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
            j +=1
        W[:,c] = W_temp
    return W
def optimization_square_loss(x_value,y_temp,W,class_number,lamda = 0):
    print('Training coordinate descent initiate using the square loss')
    n_number, project_d = x_value.shape
    #original optimization method
    x_value = np.asmatrix(x_value)
    lamda = lamda/project_d
    for c in range(class_number):
        # print(c)
        W_temp = np.asmatrix(W[:,c])
        y_temp_c = np.asmatrix(y_temp[:,c]).reshape(-1,1)
        # print(y_temp_c.shape,x_value.shape,W_temp.shape)
        temp_store = x_value.dot(W_temp)
        # print(temp_store.shape,y_temp_c.shape)
        # init= 1-np.multiply(y_temp_c,temp_store)
        squre_loss = np.sum(np.clip(init,0,None))/n_number
        regularization = np.count_nonzero(W_temp)*lamda
        loss_new = np.sum(hinge_loss)+ regularization
        loss_old =  2*loss_new
        j = 0
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
                    derta = init - temp # change to 1
                    regularization += lamda
                    derta2 = init + temp # change to -1

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
            j +=1
        W[:,c] = W_temp
    return W
def binary_optimization(x_value,y_temp,W,class_number,lamda):
    n_number, project_d = x_value.shape
    #original optimization method
    x_value = np.asmatrix(x_value)
    lamda = lamda / project_d
    for c in range(class_number):
        W_temp = np.asmatrix(W[:, c])
        y_temp_c = np.asmatrix(y_temp[:, c]).reshape(-1, 1)
        temp_store = x_value.dot(W_temp)
        init = 1 - np.multiply(y_temp_c, temp_store)
        hinge_loss = np.sum(np.clip(init, 0, None)) / n_number
        regularization = np.count_nonzero(W_temp) * lamda
        loss_new = np.sum(hinge_loss)
        loss_old = 2 * loss_new
        while (loss_old-loss_new)/loss_old >= 1e-6:
            loss_old = loss_new
            for i in np.random.choice(project_d, project_d, replace=False):
                if W_temp[i] != 0:
                    w_i = W_temp[i]
                    w_i2 = -W_temp[i]
                    temp = np.multiply(y_temp_c,x_value[:,i]*w_i)
                    derta2 = init + 2 * temp
                    loss = np.sum(np.clip(derta2,0,None))/n_number+regularization
                    if loss < loss_new:
                        loss_new = loss
                        init = derta2
                        W_temp[i] = w_i2
        W[:,c] = W_temp
    return W




def compare_init_optimization(x_value,y_temp,W,class_number,lamda = 0):
    # print('Training coordinate descent initiate from result of linear svm')
    n_number, project_d = x_value.shape
    #original optimization method
    loss_result = []
    x_value = np.asmatrix(x_value)
    lamda = lamda/project_d
    for c in range(class_number):
        W_temp = np.asmatrix(W[:,c])
        y_temp_c = np.asmatrix(y_temp[:,c]).T
        # print(y_temp_c.shape,x_value.shape,W_temp.shape)
        temp_store = x_value.dot(W_temp)
        # print(temp_store.shape,y_temp_c.shape)
        init= 1-np.multiply(y_temp_c,temp_store)
        hinge_loss = np.sum(np.clip(init,0,None))/n_number
        regularization = np.count_nonzero(W_temp)*lamda
        loss_new = np.sum(hinge_loss)+ regularization
        loss_old =  2*loss_new
        j = 0
        loss_result.append(loss_new)
        while (loss_old - loss_new) / loss_old >= 1e-6:
            loss_old = loss_new
            # for i in np.random.choice(project_d, project_d, replace=False):
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
                    derta = init - temp # change to 1
                    regularization += lamda
                    derta2 = init + temp # change to -1

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
            # print(np.count_nonzero(W_temp))
            j +=1
            loss_result.append(loss_new)
        W[:,c] = W_temp
    return W,loss_result



def predict_acc(x_value,y_temp,test_x,test_y,W,class_number):
    clf2 = LinearSVC(dual = False)
    # print(type(x_value),type(W))
    # print(np.dot(x_value, W).shape, y_temp.shape)
    clf2.fit(np.dot(x_value, W), y_temp)
    acc = clf2.score(np.dot(test_x, W), test_y)
    # if class_number != 1:
    #     predict = np.argmax(np.array(np.dot(x_value, W)), axis=1)
    #     y_lable = np.argmax(y_temp, axis=1)
    #     acc = accuracy_score(np.array(y_lable), np.array(predict))
    # else:
    #     predict = np.array(np.dot(x_value, W))
    #     acc = accuracy_score(np.sign(y_temp), np.sign(predict))
    return  acc