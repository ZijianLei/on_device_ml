from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import numpy as np
from extra_function import *
import sklearn

from sklearn.kernel_approximation import *

def main(name):
    acc_1 = []
    acc_2 = []
    acc_3 = []
    print('start')
    '''
    read data and parameter initialize for the hadamard transform
    '''
    iteration = 1
    acc_binary_2 = []
    p = FLAGS.p
    for iter in range(iteration):
        x_train, y_train, x_test, y_test = get_data(name, FLAGS)


        print('Training')
        x, y = x_train, y_train
        class_number = len(np.unique(y))
        if class_number == 2:
            class_number -= 1
        n_number, f_num = np.shape(x)
        d = 2 ** math.ceil(np.log2(f_num))
        sigma = FLAGS.s / (p * d)
        FLAGS.data_number = n_number
        clf2 = LinearSVC(dual = False)
        rbf_feature = RBFSampler(gamma=sigma, random_state=1, n_components=p * d)
        sampler = rbf_feature.fit(x)
        x_value_rks = np.sign(sampler.transform(x))
        clf2.fit(x_value_rks, y)
        '''
        Start the test process
        '''
        x, y = x_test, y_test
        n_number, f_num = np.shape(x)
        FLAGS.data_number = n_number
        test_x_rks = np.sign(sampler.transform(x))
        acc_binary_2.append(clf2.score(test_x_rks, y))
        print('accuracy of the BCSIK embedding with linear svm', acc_binary_2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser.add_argument('-p', type=int,
                        default=None,
                        help='number of the different feature_binarizing random projection')
    parser.add_argument('-data_number', type=int,
                        default=None,
                        help='number of data (no need to initialize)')

    parser.add_argument('-s', type=float,
                        default=1,
                        help='sigma of random Gaussian distribution ')

    parser.add_argument('-d_openml', type=int,
                        default=None,
                        help='data is download from openml\
                        available data:\
                        0:Fashion-MNIST,\
                        1:mnist_784'
                        )

    '''
        available data:
        0:Fashion-MNIST,
        1:mnist_784
    '''
    parser.add_argument('-d_libsvm', type=str,
                        default=None,
                        help='using data from libsvm dataset with data is well preprocessed')


    np.set_printoptions(threshold=np.inf, suppress=True)
    FLAGS, unparsed = parser.parse_known_args()
    name_space = ['Fashion-MNIST', 'mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    print(name)
    main(name)
