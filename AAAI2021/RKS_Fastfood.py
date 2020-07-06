from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
from extra_function import *
from sklearn.kernel_approximation import *



def main(name ):
    print('start')
    '''
    read data and parameter initialize for the hadamard transform
    '''
    iteration = 1
    p = FLAGS.p
    for iter in range(iteration):
        x_train, y_train, x_test, y_test = get_data(name, FLAGS)
        x,y = x_train,y_train
        n_number, f_num = np.shape(x)
        d = 2 ** math.ceil(np.log2(f_num))
        PI_value, G, B, S = parameters(n_number, p, d, FLAGS)
        FLAGS.data_number = n_number
        class_number = len(np.unique(y))
        if class_number == 2:
            class_number -= 1
        '''
        Start to prossssscessing the label
        '''
        # if method !=0:
        FLAGS.b = np.random.uniform(0, 2 * math.pi, d * p)
        FLAGS.t = np.random.uniform(-1, 1, d * p)
        sigma = FLAGS.s/(p*d)
        print('Training Linear SVM on fastfood kernel approximation')
        x, y = x_train, y_train
        n_number, f_num = np.shape(x)
        FLAGS.data_number = n_number
        x_value = np.asmatrix(fastfood(d, f_num, x, G, B, PI_value, S,FLAGS,sigma))
        clf = LinearSVC(dual = False)
        clf.fit(x_value, y)
        print('Training Linear SVM on RKS kernel approximation')
        clf2 = LinearSVC(dual = False)
        rbf_feature = RBFSampler(gamma=sigma,random_state=1,n_components=p*d)
        sampler = rbf_feature.fit(x)
        x_value_rks = sampler.transform(x)
        clf2.fit(x_value_rks, y)
        '''
        Start the test process
        '''
        x, y = x_test, y_test
        n_number, f_num = np.shape(x)
        FLAGS.data_number = n_number
        test_x = np.asmatrix(fastfood(d, f_num, x, G, B, PI_value, S,FLAGS,sigma))
        test_x_rks = sampler.transform(x)
        acc_fastfood = clf.score(test_x,y)
        acc_rks = clf2.score(test_x_rks,y)
        print('predict accuracy of RKS: %f' %(acc_rks))
        print('predict accuracy of Fastfood: %f' %(acc_fastfood))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser.add_argument('-p', type=int,
                        default=None,
                        help='number of the different hadamard random projection')
    parser.add_argument('-data_number', type=int,
                        default=1,
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
    name_space = ['Fashion-MNIST','mnist_784']
    if FLAGS.d_openml != None:
        name = name_space[FLAGS.d_openml]
    else:
        name = FLAGS.d_libsvm
    print(name)
    main(name)
