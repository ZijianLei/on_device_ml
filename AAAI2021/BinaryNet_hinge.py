from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
import numpy as np
import copy
import scipy
from extra_function import *
from imblearn.over_sampling import SMOTE
import ffht
from imblearn.over_sampling import RandomOverSampler
import sklearn
from collections import Counter
import matplotlib.pyplot as plt
import torch
class Binarynet:
    def __init__(self,PI_value, G, B, S,d ,class_number):
        self.G = G
        self.B = B
        self.PI_value = PI_value
        self.unpermutate =np.argsort(self.PI_value) # used in backporpagation
        self.S = S/(FLAGS.s*np.sqrt(FLAGS.p * d) ** 3)
        self.dimension = d
        self.class_number = class_number
        self.full_data = None
        self.full_label = None
        self.full_data_test = None
        self.full_label_test = None
        self.batch = None #data within a batch
        self.label = None
        self.coefficients = np.random.randn(self.dimension*FLAGS.p,self.class_number)
        self.coefficients_temp = copy.deepcopy(self.coefficients)
        #coeeficients shape is (d_project,class_number)torch.randn((self.dimension*FLAGS.p,self.class_number),requires_grad = True)
        self.loss = None # the objective function
        self.lr = 1e-1
        self.epoch = 100
        self.gradient = None
        self.prediction = None
        self.wb = None
        self.x_transfer = None
        self.x_S = None
        self.x_G = None
        self.x_B = None
        self.momentum = 0.9
        self.momentum_W = 0
        self.momentum_S = 0
        self.momentum_G = 0
        self.plt_train_acc = []
        self.plt_test_acc = []
        self.plt_train_loss = []
        self.plt_test_loss = []
        self.ywx = None

    def Forwardpropogation(self):
        p = FLAGS.p
        x_ = self.batch

        n = x_.shape[0]
        x_ = np.pad(x_, ((0, 0), (0, self.dimension - x_.shape[1])), 'constant', constant_values=(0, 0))
        x_ = np.tile(x_, (1, p))
        self.original = x_
        self.x_B = np.multiply(x_, self.B)
        x_ = self.x_B.reshape(n * p, self.dimension)

        for i in range(n * p):

            ffht.fht(x_[i])
        x_ = x_.reshape(n, p * self.dimension)
        self.x_G = np.take(x_, self.PI_value, axis=1, mode='wrap')
        x_transformed = np.multiply(self.x_G, self.G)
        x_ = np.reshape(x_transformed, (n * p, self.dimension))
        for i in range(n * p):
            ffht.fht(x_[i])
        self.x_S = x_.reshape(n, p * self.dimension)
        x_= np.multiply(self.x_S, self.S)

        # self.x_transfer = np.sign(x_)
        self.x_transfer = copy.deepcopy(x_)
        # print(np.max(self.x_transfer),np.mean(self.x_transfer),np.min(self.x_transfer))
        # exit()
        x_transfer= np.cos(x_+ FLAGS.b) + FLAGS.t
        x_= np.sign(x_transfer)
        self.wb = np.sign(self.coefficients)
        '''
        using the sigmoid function we will use the sigmoid activation for the multiclass problem
        '''
        prediction = x_.dot(np.sign(self.coefficients))
        self.ywx = np.multiply(self.label, prediction)
        self.prediction = np.where(prediction>0,1,-1)

        self.loss = np.sum(np.clip(1-self.ywx, 0, None)) / x_.shape[0]
        # self.loss = sklearn.metrics.log_loss(self.label, prediction)
        self.ywx = np.where(self.ywx<1,1,0)

        self.gradient = - np.multiply(np.mean(self.ywx,axis=0),np.dot(x_.T,  self.label))
        # print(np.count_nonzero(self.gradient))
        # print(self.gradient[0,:10])
        # print(self.coefficients[0,:10])
        '''
        using the softmax function
        '''
        # predict_temp = x_.dot(np.sign(self.coefficients))
        # # predict_temp -= np.max(predict_temp)
        # prediction = scipy.special.softmax(predict_temp,axis = 1)
        # self.prediction = prediction
        # self.loss = -np.sum(np.multiply(self.label,np.log(prediction)))/x_.shape[0]
        # self.gradient = np.dot(x_.T, prediction - self.label)/x_.shape[0]

        # accuracy = sklearn.metrics.accuracy_score(np.argmax(self.prediction, axis=1), np.argmax(self.label, axis=1))


    def STE_layer(self):
        return np.multiply(self.gradient,np.where(abs(self.coefficients)<=1,1,0))

    def Backpropogation(self):
        p = FLAGS.p
        # self.gradient = self.STE_layer()
        self.coefficients -= self.update_value()  #updata the W
    def Fastfood_updata(self):
        p = FLAGS.p
        # print(self.label.shape)
        self.gradient  =- np.multiply(np.mean(self.ywx,axis=1).reshape(-1,1),np.dot((self.label)/self.original.shape[0],self.wb.T))
        # print(self.gradient.shape)
        self.gradient = np.multiply(self.gradient,-np.sin(self.x_transfer + FLAGS.b))
        self.gradient = np.array(self.gradient)

        S = copy.deepcopy(self.S)
        self.S -= self.update_S()
        self.gradient= np.multiply(S,self.gradient)  # gradient update after S

        self.gradient = self.gradient.reshape(-1,self.dimension)
        for i in range(self.gradient.shape[0]):
            ffht.fht(self.gradient[i])
        # self.gradient = self.gradient/(2**(self.dimension/2))# gradient update after H
        self.gradient = self.gradient.reshape(-1, p*self.dimension)
        G = copy.deepcopy(self.G)
        self.G -= self.update_G()
        # print(np.mean(S),np.mean(G))
        # self.gradient = np.multiply(G, self.gradient)# gradient update after g
        # np.take(self.gradient, self.unpermutate, axis=1, out=self.gradient) #gradient update after Pi
        # self.gradient = self.gradient.reshape(-1,self.dimension)
        # for i in range(self.gradient.shape[0]):
        #     ffht.fht(self.gradient[i])
        # self.gradient = self.gradient/(2**(self.dimension/2)) # gradient update after H
        # self.gradient = self.gradient.reshape(-1, p*self.dimension)
        # # self.B = np.sign(self.B - self.update_B())
        # self.gradient = np.multiply(self.B, self.gradient) # gradient updata after B if we have further layer in the deep neural network
    def update_S(self):
        self.momentum_S = self.momentum*self.momentum_S+self.lr *np.diag(np.dot(self.x_S.T, self.gradient))*1e-5
        return  self.momentum_S
        # return self.lr * np.diag(np.dot(self.x_S,self.gradient))
    def update_G(self):
        self.momentum_G = self.momentum * self.momentum_G + self.lr * np.diag(np.dot(self.x_G.T, self.gradient))*1e-5
        return self.momentum_G
        # return self.lr * np.diag(np.dot(self.x_G.T, self.gradient))
    def update_B(self):
        return  self.lr *np.diag(np.dot(self.original.T, self.gradient))
        # return self.lr*np.diag(np.dot(self.original,self.gradient))



    def update_value(self):
        self.momentum_W = self.momentum *self.momentum_W+self.lr*self.gradient
        return self.momentum_W
    def train(self):
        for i in range(self.epoch):
            self.batch = self.full_data
            self.label = self.full_label
            acc = self.predict()
            self.plt_train_acc.append(acc)
            self.plt_train_loss.append(self.loss)
            temp = np.hstack((self.full_label,self.full_data))
            rng = np.random.default_rng()
            rng.shuffle(temp)
            for example in np.array_split(temp,np.int(temp.shape[0]/FLAGS.batch_size)):
                self.label = example[:,:self.class_number]
                self.batch = example[:,self.class_number:]
                self.Forwardpropogation()
                self.Backpropogation()
                self.Fastfood_updata()
            self.label = self.full_label_test
            self.batch = self.full_data_test
            acc = self.predict()
            self.plt_test_acc.append(acc)
            self.plt_test_loss.append(self.loss)
            # print(acc)
            # if i!=0 and i%3== 0:
            #     self.lr /= 2
    def predict(self):
        # self.batch = self.full_data
        # self.label = self.full_label
        self.Forwardpropogation()

        # print(self.label)
        # print(np.argmax(self.prediction, axis=1)[:10], np.argmax(self.label, axis=1)[:10])
        # print(self.prediction[:2])
        accuracy = sklearn.metrics.accuracy_score(np.argmax(self.prediction,axis = 1), np.argmax(self.label,axis = 1))
        return accuracy


def main(name ):
    print('start')
    '''
    read data and parameter initialize for the hadamard transform
    '''
    p = FLAGS.p
    lamb = FLAGS.l
    x_train, y_train, x_test, y_test = get_data(name, FLAGS)
    x,y = x_train,y_train
    n_number, f_num = np.shape(x)
    d = 2 ** math.ceil(np.log2(f_num))
    FLAGS.data_number = n_number
    class_number = len(np.unique(y))
    loss = []
    y_temp = label_processing_hinge(y, n_number, FLAGS)
    # if class_number == 2:
    #     class_number -= 1
    PI_value, G_0, B, S_0 = parameters(n_number,p,d,FLAGS)
    BNet = Binarynet(PI_value, G_0, B, S_0,d ,class_number)

    BNet.full_data = x
    BNet.full_label = y_temp

    BNet.class_number = class_number
    n_number, f_num = np.shape(x_test)
    FLAGS.data_number = n_number
    y_temp_test = label_processing_hinge(y_test, n_number, FLAGS)
    BNet.full_data_test = x_test
    BNet.full_label_test = y_temp_test
    BNet.train()
    print('the predict result (adaptive fastfood)', BNet.predict())
    # plt.plot(np.arange(len(BNet.plt_train_acc[1:])),BNet.plt_train_acc[1:],label = 'train_fastfood')
    # plt.plot(np.arange(len(BNet.plt_test_acc)), BNet.plt_test_acc,label ='test_fastfood')
    # # the result only compute the STE of coefficient
    # train_loss = copy.deepcopy(BNet.plt_train_loss)
    # test_loss = copy.deepcopy(BNet.plt_test_loss)
    # BNet.G = G_0
    # BNet.B = B
    # BNet.S = S_0/(FLAGS.s*np.sqrt(FLAGS.p * d) ** 3)
    # BNet.lr = 1e-2
    # BNet.plt_test_acc = []
    # BNet.plt_train_acc = []
    # BNet.plt_test_loss = []
    # BNet.plt_train_loss = []
    # BNet.train_coefficient()
    # plt.plot(np.arange(len(BNet.plt_train_acc[1:])), BNet.plt_train_acc[1:], label='train_w')
    # plt.plot(np.arange(len(BNet.plt_test_acc)), BNet.plt_test_acc, label='test_w')
    # plt.legend(prop={'size': 15})
    # plt.savefig('%s_%f_acc.jpg' % (name,FLAGS.s))
    # plt.show()
    # plt.cla()
    # plt.plot(np.arange(len(train_loss[1:])), train_loss[1:], label='train_fastfood')
    # plt.plot(np.arange(len(test_loss)), test_loss, label='test_fastfood')
    # plt.plot(np.arange(len(BNet.plt_train_loss[1:])), BNet.plt_train_loss[1:], label='train_w')
    # plt.plot(np.arange(len(BNet.plt_test_loss)), BNet.plt_test_loss, label='test_w')
    # plt.legend(prop={'size': 15})
    # plt.show()
    # plt.savefig('%s_%f_loss.jpg' % (name, FLAGS.s))
    # exit()


    print('the predict result',BNet.predict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser.add_argument('-p', type=int,
                        default=8,
                        help='number of the different feature_binarizing random projection,\
                        the final dimension is p*d, d is the original dimension'
                             )
    parser.add_argument('-batch_size', type=int,
                        default=256,
                        help='number of data')
    parser.add_argument('-data_number', type=int,
                        default=None,
                        help='number of data')
    parser.add_argument('-b', type=float,
                        default=None,
                        help='Parameters for Feature embedding sign(cos(Rx+b)+t)')
    parser.add_argument('-t', type=float,
                        default=None,
                        help='Parameters for Feature embedding sign(cos(Rx+b)+t)')
    parser.add_argument('-s', type=float,
                        default=1,
                        help='sigma of random Gaussian distribution ')
    parser.add_argument('-l', type=float,
                        default=0,
                        help='the regularization parameter ')

    parser.add_argument('-d_openml', type=int,
                        default=None,
                        help='data is download from openml\
                        available data:\
                        0:Fashion-MNIST,\
                        1:mnist_784'
                             )

    parser.add_argument('-d_libsvm', type=str,
                        default=None,
                        help='using data from libsvm dataset, you can download the original data from\
                        LIBSVM Dataset. We have provided usps dataset')

    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name_space = ['Fashion-MNIST','mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    print(name)
    main(name)
