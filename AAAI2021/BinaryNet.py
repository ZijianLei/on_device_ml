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
        self.S = S
        self.dimension = d
        self.class_number = class_number
        self.full_data = None
        self.full_label = None
        self.batch = None #data within a batch
        self.label = None
        self.coefficients = np.random.randn(self.dimension*FLAGS.p,self.class_number)
        #coeeficients shape is (d_project,class_number)torch.randn((self.dimension*FLAGS.p,self.class_number),requires_grad = True)
        self.loss = None # the objective function
        self.lr = 1e-1
        self.epoch = 5
        self.gradient = None
        self.prediction = None
        self.wb = None
        self.x_transfer = None
        self.x_S = None
        self.x_G = None
        self.x_B = None
        self.plt = []

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
        x_= x_/(np.sqrt(p * self.dimension) ** 3)
        # self.x_transfer = np.sign(x_)
        self.x_transfer = copy.deepcopy(x_)
        x_transfer= np.cos(x_+ FLAGS.b) + FLAGS.t
        x_= np.sign(x_transfer)
        self.wb = np.sign(self.coefficients)
        '''
        using the sigmoid function we will use the sigmoid activation for the multiclass problem
        '''
        prediction = 1/(1+np.exp(-x_.dot(np.sign(self.coefficients))) )
        self.prediction = prediction
        self.loss = sklearn.metrics.log_loss(self.label, prediction)
        self.gradient = np.dot(x_.T, prediction - self.label) / x_.shape[0]
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
        self.gradient = self.STE_layer()
        # self.coefficients = np.clip(self.coefficients,-1,1)
        self.coefficients -= self.update_value()  #updata the W
    def Fastfood_updata(self):
        p = FLAGS.p
        self.gradient  = np.dot((self.prediction-self.label)/self.original.shape[0],self.wb.T)
        self.gradient = np.multiply(self.gradient,-np.sin(self.x_transfer + FLAGS.b))
        self.gradient = np.array(self.gradient)
        # print(self.gradient)
        # exit()

        S = copy.deepcopy(self.S)
        # print(self.update_S())
        self.S -= self.update_S()
        # exit()
        # print(S)
        # # print(S-self.update_S())
        # exit()
        self.gradient= np.multiply(S,self.gradient)  # gradient update after S
        #
        # self.gradient = self.gradient.reshape(-1,self.dimension)
        # for i in range(self.gradient.shape[0]):
        #     ffht.fht(self.gradient[i])
        # self.gradient = self.gradient/(2**(self.dimension/2))# gradient update after H
        # self.gradient = self.gradient.reshape(-1, p*self.dimension)
        # G = copy.deepcopy(self.G)
        # self.G -= self.update_G()
        # self.gradient = np.multiply(G, self.gradient)# gradient update after g
        # np.take(self.gradient, self.unpermutate, axis=1, out=self.gradient) #gradient update after Pi
        # self.gradient = self.gradient.reshape(-1,self.dimension)
        # for i in range(self.gradient.shape[0]):
        #     ffht.fht(self.gradient[i])
        # self.gradient = self.gradient/(2**(self.dimension/2)) # gradient update after H
        #self.gradient = self.gradient.reshape(-1, p*self.dimension)
        # # self.B = np.sign(self.B - self.update_B())
        # self.gradient = np.multiply(self.B, self.gradient) # gradient updata after B if we have further layer in the deep neural network
    def update_S(self):
        return  self.lr *np.diag(np.dot(self.x_S.T, self.gradient))
        # return self.lr * np.diag(np.dot(self.x_S,self.gradient))
    def update_G(self):
        return self.lr * np.diag(np.dot(self.x_G.T, self.gradient))
        # return self.lr * np.diag(np.dot(self.x_G,self.gradient ))
    def update_B(self):
        return  self.lr *np.diag(np.dot(self.original.T, self.gradient))
        # return self.lr*np.diag(np.dot(self.original,self.gradient))

    def update_value(self):
        return self.lr*self.gradient

    def train(self):
        for i in range(self.epoch):
            temp = np.hstack((self.full_label,self.full_data))
            rng = np.random.default_rng()
            # self.label = temp[:, :self.class_number]
            # self.batch = temp[:, self.class_number:]
            rng.shuffle(temp)
            for example in np.array_split(temp,np.int(temp.shape[0])):
                self.label = example[:,:self.class_number]
                self.batch = example[:,self.class_number:]

                self.Forwardpropogation()
                self.Backpropogation()
                # if i!=0 and i/18 == 0:
                self.Fastfood_updata()
            self.lr = 1/(i+1)

            # self.label = self.full_label
            # self.batch = self.full_data
            # self.Forwardpropogation()
            # self.Backpropogation()
            acc = self.predict()
            print(i,acc)
            # if i!=0 and i%3== 0:
            #     self.lr /= 2


    def predict(self):
        self.batch = self.full_data
        self.label = self.full_label
        self.Forwardpropogation()

        # print(self.label)
        # print(np.argmax(self.prediction, axis=1)[:100], np.argmax(self.label, axis=1)[:100])
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
    y_temp = label_processing(y, n_number, FLAGS)
    if class_number == 2:
        class_number -= 1
    PI_value, G, B, S = parameters(n_number,p,d,FLAGS)
    BNet = Binarynet(PI_value, G, B, S,d ,class_number)

    BNet.full_data = x
    BNet.full_label = y_temp

    BNet.class_number = class_number
    BNet.train()
    # plt.plot(np.arange(50),BNet.plt)
    # plt.show()
    # exit()
    x, y = x_test, y_test
    n_number, f_num = np.shape(x)
    FLAGS.data_number = n_number
    y_temp = label_processing(y, n_number, FLAGS)

    BNet.full_data = x
    BNet.full_label = y_temp
    print('the predict result',BNet.predict())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser.add_argument('-p', type=int,
                        default=8,
                        help='number of the different feature_binarizing random projection,\
                        the final dimension is p*d, d is the original dimension'
                             )
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
