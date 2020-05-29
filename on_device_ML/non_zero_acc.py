import numpy as np
import argparse
import matplotlib.pyplot as plt
def main(name):
    acc_1 = []
    acc_2 = []
    acc_3 = []
    non_2 = []
    acc_fastfood = []
    acc_rks = []
    acc_Fastfood = np.load('%s_Fastfood.npy' % name ).reshape(3,-1)
    acc_RKS = np.load('%s_RKS.npy' % name ).reshape(3,-1)
    if 1:
        acc_binary_1 = np.load('%s_sign.npy' % name )

        acc_binary_2 = np.load('%s_0_init.npy' % name )
        acc_binary_3 = np.load('%s_svm_cos_similarity.npy' % name )
        non_zero_1 = np.load('%s_non0_sign.npy' % name)
        print(np.shape(acc_binary_1))
        non_zero_2 = np.load('%s_non0_0_init.npy' % name)
        non_zero_3 = np.load('%s_non0_cos_similarity.npy' % name)

        # print(acc_binary_3)
        for T in range(6):
            acc_2 = []
            non_2 = []
            # acc_1.append(np.max(np.mean(acc_binary_1[T].reshape(3,-1),axis=0)))
            temp = acc_binary_1[T].reshape(-1, 4)
            temp_non0 = non_zero_1[T].reshape(-1, 4)
            max_sigma = np.argmax(np.mean( temp,axis=0))
            # max_sigma = 3
            acc_2.append(np.mean(temp[:,max_sigma].reshape(3,-1),axis=0))
            non_2.append(np.mean(temp_non0[:,max_sigma].reshape(3,-1),axis=0))
            # acc_3.append(np.max(np.mean(acc_binary_3[T].reshape(3, -1), axis=0 )))
            acc_2 = np.array(acc_2)
            non_2 = np.array(non_2)
            print(acc_2.shape)
            plt.scatter(np.array(non_2)[0,:],np.array(acc_2)[0,:],linewidth=2,linestyle= '--',label = 'T = %d' %(T+1))
    plt.legend()
    plt.ylabel('accuracy')
    plt.xlabel('number of non-zero coefficient')
    plt.xscale('log',basex = 2)
    plt.savefig('non_zero.png')
    plt.show()
        # print(acc_1,acc_2,acc_3)
    # acc_rks.append(np.max(np.mean(acc_RKS,axis = 0)))
    # acc_fastfood.append(np.max(np.mean(acc_Fastfood, axis=0)))
    # print(acc_fastfood,acc_rks)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str,
                        default=None,
                        help='name of dataset')

    FLAGS, unparsed = parser.parse_known_args()
    name = FLAGS.d
    print(name)
    main(name)