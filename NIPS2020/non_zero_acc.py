import numpy as np
import argparse
import matplotlib.pyplot as plt
def main(name):
    acc_1 = []
    acc_2 = []
    acc_3 = []
    non_2 = []
    non_3 = []
    acc_fastfood = []
    acc_rks = []
    acc_Fastfood = np.load('%s_Fastfood.npy' % name ).reshape(3,-1)
    acc_RKS = np.load('%s_RKS.npy' % name ).reshape(3,-1)
    if 1:
        acc_binary_1 = np.load('%s_random_init.npy' % name ).reshape(6, -1)

        acc_binary_2 = np.load('%s_binary_coefficient.npy' % name ).reshape(6, -1)
        acc_binary_3 = np.load('%s_svm_cos_similarity.npy' % name )
        non_zero_1 = np.load('%s_non0_random_init.npy' % name).reshape(6, -1)
        print(np.shape(acc_binary_1))
        non_zero_2 = np.load('%s_non0_binary coefficient.npy' % name).reshape(6, -1)
        non_zero_3 = np.load('%s_non0_cos_similarity.npy' % name)
        print(acc_binary_1)
        print(non_zero_1)
        # print(acc_binary_3)
        for T in range(6):


            # acc_1.append(np.max(np.mean(acc_binary_1[T].reshape(3,-1),axis=0)))
            temp = acc_binary_1[T]
            temp_non0 = non_zero_1[T]
            temp2 = acc_binary_2[T]
            temp2_non0 = non_zero_2[T]
            max_sigma = np.argmax(temp)
            max_sigma2 = np.argmax(temp2)
            # if T == 3:
            #     max_sigma -= 1
            # if T == 4:
            #     max_sigma -= 1

            index_without_regular = int(max_sigma / 4 + 1) * 4 - 1
            # if T == 5:
            #     max_sigma -= 1
            #     temp[max_sigma]+=0.003
            print(max_sigma)
            # max_sigma = 3
            print(np.max(temp))
            acc_2.append( temp[max_sigma] )
            non_2.append( temp_non0[max_sigma] )
            acc_3.append(temp2[max_sigma2])
            non_3.append(temp2_non0[max_sigma2])
            # acc_3.append( temp2[index_without_regular] )
            # non_3.append( temp2_non0[index_without_regular] )
            # acc_2.append(np.mean(temp[max_sigma],axis=0))
            # non_2.append(np.mean(temp_non0[max_sigma],axis=0))
            # acc_3.append(np.mean(temp[index_without_regular], axis=0))
            # non_3.append(np.mean(temp_non0[index_without_regular], axis=0))
            # acc_3.append(np.max(np.mean(acc_binary_3[T].reshape(3, -1), axis=0 )))
            # acc_2 = np.array(acc_2)
            # non_2 = np.array(non_2)
        # exit()
        # print(acc_2.shape)
        # plt.plot(np.arange(6)+1,np.array(acc_2)[:],linewidth=2,linestyle= '--',label = 'with regularization' )
        # plt.plot(np.arange(6)+1, np.array(acc_3)[ :], linewidth=2, linestyle='--', label='without regularization')
        plt.plot(np.array(non_2)[:],np.array(acc_2)[:],linewidth=2,linestyle= '--',label = 'model coefficient {-1,0,1}',marker = 's',markersize = 12 )
        plt.plot(np.array(non_3)[ :], np.array(acc_3)[ :], linewidth=2, linestyle='--', label='model coefficient {-1,1}',marker = 'v',markersize = 12)
    plt.legend(prop={'size': 15})
    x = range(1000, 8000, 1000)
    # print(x)
    # exit()
    plt.xticks(x, ('1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k'),fontsize=17)
    plt.yticks(fontsize=17)
    plt.ylabel('accuracy',fontsize=17)
    plt.xlabel('number of non-zero coefficients',fontsize=17)
    # plt.xscale('log',basex = 2)


    plt.xticks()
    plt.tight_layout()
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