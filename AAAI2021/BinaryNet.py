from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
import numpy as np
from extra_function import *
from imblearn.over_sampling import SMOTE
import ffht
from imblearn.over_sampling import RandomOverSampler
import sklearn
from collections import Counter
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
        self.batch = None #data within a batch
        self.label = None
        self.coefficients = np.random.randn(self.dimension*FLAGS.p,self.class_number)
        #coeeficients shape is (d_project,class_number)torch.randn((self.dimension*FLAGS.p,self.class_number),requires_grad = True)
        self.loss = None # the objective function
        self.lr = 4e-1
        self.epoch = 1000
        self.gradient = None
        self.prediction = None
        self.wb = None
        self.x_transfer = None

    def Forwardpropogation(self):
        p = FLAGS.p
        x_ = self.batch
        n = x_.shape[0]
        x_ = np.pad(x_, ((0, 0), (0, self.dimension - x_.shape[1])), 'constant', constant_values=(0, 0))
        x_ = np.tile(x_, (1, p))
        self.original = x_
        x_i = np.multiply(x_, self.B)
        x_ = x_i.reshape(n * p, self.dimension)
        for i in range(n * p):
            ffht.fht(x_[i])
        x_ = x_.reshape(n, p * self.dimension)
        np.take(x_, self.PI_value, axis=1, mode='wrap', out=x_)
        x_transformed = np.multiply(x_, self.G)
        x_ = np.reshape(x_transformed, (n * p, self.dimension))
        for i in range(n * p):
            ffht.fht(x_[i])
        x_ = x_.reshape(n, p * self.dimension)
        x_= np.multiply(x_, self.S)
        x_= x_/(np.sqrt(p * self.dimension) ** 3)
        self.x_transfer= np.cos(x_+ FLAGS.b) + FLAGS.t
        x_= np.sign(self.x_transfer)
        self.wb = np.sign(self.coefficients)
        prediction = 1/(1+np.exp(-x_.dot(np.sign(self.coefficients))) )
        # we will use the sigmoid activation for the multiclass problem
        self.prediction = prediction
        self.loss = sklearn.metrics.log_loss(self.label,prediction)
        self.gradient = np.dot(x_.T,prediction-self.label)/x_.shape[0]
        # print(self.coefficients)
        print(np.argmax(self.prediction, axis=1)[:10], np.argmax(self.label, axis=1)[:10])
        accuracy = sklearn.metrics.accuracy_score(np.argmax(self.prediction, axis=1), np.argmax(self.label, axis=1))
        print(accuracy,self.loss)

    def STE_layer(self):
        return np.clip(self.gradient,-1,1)

    def Backpropogation(self):
        p = FLAGS.p
        self.gradient = self.STE_layer()
        # self.coefficients = np.clip(self.coefficients,-1,1)
        self.coefficients -= self.update_value()
        self.gradient  = np.dot((self.prediction-self.label)/self.original.shape[0],self.wb.T).T
        self.gradient = self.STE_layer().dot(-np.sin(self.x_transfer))
        self.gradient = np.diag(self.gradient)
        self.gradient= np.multiply(self.S,self.gradient) #updating S
        self.S -= self.update_value()
        self.gradient = self.gradient.reshape(p,-1)
        for i in range(p):
            ffht.fht(self.gradient[i])
        self.gradient = self.gradient.reshape( -1)/(2**(self.dimension/2))

        self.gradient = np.multiply(self.G, self.gradient)  # updating S
        self.G -= self.update_value()
        self.gradient = self.gradient.reshape(1,-1)
        np.take(self.gradient, self.unpermutate, axis=1, out=self.gradient)
        self.gradient = self.gradient.reshape(p, -1)
        for i in range(p):
            ffht.fht(self.gradient[i])
        self.gradient = self.gradient.reshape( -1)/(2**(self.dimension/2))
        self.gradient = np.multiply(self.B, self.gradient)  # updating S
        self.B = np.sign(self.B - self.update_value())

    def update_value(self):
        return self.lr*self.gradient

    def train(self):
        for i in range(self.epoch):
            self.Forwardpropogation()
            self.Backpropogation()
            if i%200 == 0:
                self.lr /= 0.5

    def predict(self):
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

    y_temp = label_processing(y, n_number, FLAGS)
    if class_number == 2:
        class_number -= 1
    PI_value, G, B, S = parameters(n_number,p,d,FLAGS)
    BNet = Binarynet(PI_value, G, B, S,d ,class_number)

    BNet.batch = x
    BNet.label = y_temp
    BNet.class_number = class_number
    BNet.train()
    x, y = x_test, y_test
    n_number, f_num = np.shape(x)
    FLAGS.data_number = n_number
    y_temp = label_processing(y, n_number, FLAGS)
    BNet.batch = x
    BNet.label = y_temp
    BNet.Forwardpropogation()
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
