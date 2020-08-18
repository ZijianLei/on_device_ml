from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import time
import numpy as np
from extra_function import *
import matplotlib.pyplot as plt
def main(name ):
    print('start')
    '''
    read data and parameter initialize for the hadamard transform
    '''
    iteration = 1
    acc_binary_linear = []
    acc_binary = []
    p = FLAGS.p
    lamb = FLAGS.l
    acc_fastfood = []
    # P = [ 8, 16, 24, 32,40,48]
    P = [2,4,6,8,10,12,16]
    # P = [1,2,3,4,5]
    P = np.array(P)
    x_train, y_train, x_test, y_test = get_data(name, FLAGS)
    for p in P:
        FLAGS.p = p
        for iter in range(iteration):

                x,y = x_train,y_train
                n_number, f_num = np.shape(x)
                d = 2 ** math.ceil(np.log2(f_num))
                FLAGS.data_number = n_number
                class_number = len(np.unique(y))
                if class_number == 2:
                    class_number -= 1
                print('generalising the transform parameters')

                PI_value, G, B, S = parameters(n_number,p,d,FLAGS)
                '''
                Start to prossssscessing the label
                '''
                sigma = FLAGS.s
                x_value = np.asmatrix(fastfood(d, f_num, x, G, B, PI_value, S, FLAGS, sigma))
                clf0 = LinearSVC(dual=False)
                clf0.fit(x_value, y)
                x_value = np.sign(x_value)
                print('training the full precision linear svm model on binary data')
                clf = LinearSVC(dual = False)
                clf.fit(x_value, y)
                # W = np.asmatrix(np.sign(np.random.randn(d * p, class_number)))

                alpha = np.linalg.norm(clf.coef_,ord=1,axis=1)

                W = np.asmatrix(np.sign(clf.coef_).T)
                y_temp = label_processing(y, n_number, FLAGS)
                print('training the biniry model')
                W = optimization(x_value, y_temp, W, class_number,alpha, lamb)
                print('Finish')
                x, y = x_test, y_test
                n_number, f_num = np.shape(x)
                FLAGS.data_number = n_number
                test_x = np.asmatrix(fastfood(d, f_num, x, G, B, PI_value, S, FLAGS, sigma))
                acc_fastfood.append(clf0.score(test_x,y))
                # print('the fastfood' ,acc)
                test_x = np.sign(test_x)
                acc_binary.append(predict_acc(x_value,y_train,test_x,y_test,W))
                # print('accuracy of the binary classifier on binary data', acc_binary)
                acc_binary_linear.append(clf.score(test_x, y_test))

                # print('accuracy of the linear_svm classifier on binary data', acc_binary_linear)
    if name == 'usps':
        class_number = 10
    plt.plot(P[:5]*d*(3*32+32+class_number*32)/np.power(2,13),acc_fastfood[:5],linewidth=3, linestyle='--', label='Fastfood',marker = 's',markersize = 12)
    print(P[:5]*d*(3*32+32+class_number*32)/np.power(2,13),acc_fastfood[:5],'Fastfood')
    plt.plot(P[:5] *d* (3 * 32 + 1 + class_number * 32)/np.power(2,13), acc_binary_linear[:5], linewidth=3, linestyle='--', label='Our Method',
             marker='v', markersize=12)
    print(P[:5] *d* (3 * 32 + 1 + class_number * 32)/np.power(2,13), acc_binary_linear[:5],'Our Method')
    plt.plot(P [1:]*d* (3 * 32 + 1 + class_number)/np.power(2,13), acc_binary[1:], linewidth=3, linestyle='--', label='Our Method-b',
             marker='o', markersize=12)
    print(P [1:]*d* (3 * 32 + 1 + class_number)/np.power(2,13), acc_binary[1:],'Our Method-b')
    plt.legend(prop={'size': 15})
    plt.yticks(fontsize=17)
    plt.ylabel('Accuracy', fontsize=17)
    plt.xlabel('Memory Cost(KB)', fontsize=17)
    plt.tight_layout()
    # plt.savefig('%s.jpg' % name)
    plt.show()

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
