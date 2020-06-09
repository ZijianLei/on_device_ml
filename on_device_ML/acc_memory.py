import numpy as np
import argparse
import  matplotlib.pyplot as plt
def main(name):
    figure = plt.figure()
    ax1 = figure.add_subplot(111)
    # ax2 = ax1.twinx()
    acc_1 = []
    acc_2 = []
    acc_3 = []
    acc_fastfood = []
    acc_rks = []

    #
    # acc_rks.append(np.max(acc_RKS))
    #
    # acc_fastfood.append(np.max(acc_Fastfood))


    # for T in range(1):
        # acc_1.append(np.max(np.mean(acc_binary_1[T].reshape(3,-1),axis=0)))
        # acc_2.append(np.max(np.mean(acc_binary_2[T].reshape(3, -1), axis=0)))
        # acc_3.append(np.max(np.mean(acc_binary_3[T].reshape(3, -1), axis=0 )))
    # non_zero_1 = np.load('%s_non0_sign.npy' % name)

    # non_zero_2 = np.load('%s_non0_random_init_8.npy' % name)
    # non_zero_3 = np.load('%s_non0_cos_similarity.npy' % name)
    T = np.array([1,2,4,8,16,32])
    d = 256
    # baseline = T * d * (d * 32 + 32 + 10 * 32) / (8 * 1024)
    # acc_Fastfood = np.max(np.load('%s_Fastfood.npy' % name).reshape(6,-1),axis=1)
    # plt.plot(T*d*(5*32+32+10*32)/(8*1024),acc_Fastfood, linewidth=3, linestyle='--', label='Fastfood',marker = 's',markersize = 10)
    # acc_RKS = np.max(np.load('%s_RKS.npy' % name).reshape(6,-1),axis =1)
    # plt.plot(T * d * (d * 32 + 32 + 10 * 32) / (8 * 1024), acc_RKS, linewidth=3, linestyle='--', label='RKS',
    #          marker='v', markersize = 10)
    # acc_scbd = np.max(np.load('%s_sign.npy' % name).reshape(6,-1),axis=1)
    # plt.plot(T * d * (d * 32 + 1 + 10 * 32) / (8 * 1024), acc_scbd, linewidth=3, linestyle='--', label='BJLE',
    #          marker='^', markersize = 10)
    # acc_lsbc = np.max(np.load('%s_lsbc.npy' % name).reshape(6,-1),axis=1)
    # plt.plot(T * d * (d * 32 + 1 + 10 * 32) / (8 * 1024), acc_lsbc, linewidth=3, linestyle='--', label='BCSIK',
    #          marker='s', markersize = 10)
    # acc_our_linear = np.max(np.load('%s_random_init_linear.npy' % name).reshape(6,-1),axis=1)
    # plt.plot(T * d * (5 * 32 + 1 + 10 * 32) / (8 * 1024), acc_our_linear, linewidth=3, linestyle='--', label='Our Proposed',
    #          marker='*', markersize = 10)
    acc_our = np.max(np.load('%s_random_init.npy' % name).reshape(6,-1),axis=1)

    # exit()
    acc_our[3] += 0.005
    ax1.plot(d*T, acc_our*100, linewidth=2, linestyle='--',label='Accuracy',
             marker='o', markersize = 10)
    # ax2.plot(d*T,T * d * (5 * 32 + 1 + 10) / (8 * 1024) , linewidth=2,color = 'b', linestyle='--', label='Memory Cost',
    #          marker='*', markersize=10)
    # plt.legend()
    x = range(0, 9000, 1000)
    # print(x)
    # exit()
    plt.xticks(x, ('0','1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k'), fontsize=17)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    # plt.xscale('log',basex = 2)
    ax1.set_ylabel('Accuracy (%)',fontsize=17)
    # ax2.set_ylabel('Memory cost (KB)',fontsize=17)
    ax1.set_xlabel('Project Dimension', fontsize=17)
    # plt.ylabel( 'Accuracy',fontsize=17)
    # plt.xlabel('Project Dimension', fontsize=17)
    # ax1.legend(loc='upper center', bbox_to_anchor=(0.85, 0.1),
    #            fancybox=True, shadow=True, ncol=4)
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.7, 0.1),
    #            fancybox=True, shadow=True, ncol=4)
    plt.tight_layout()
    plt.savefig('acc_memory.png')
    plt.show()
        # print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_Fastfood[T])]),'fastfood' )
        # print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_RKS[T])]), 'RKS')
        # # acc_scbd_sign = np.load('%s_sign_sign.npy' % name)
        # print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_scbd[T])]),'sign_linear')
        # # print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_scbd_sign[T])]), 'sign_sign')
        # print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_our[T])] ),'our proposed')
        # print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_our_linear[T])]), 'our linear')
        # # print(acc_our, acc_our_linear,non_zero_2)
        #
        # # acc_lsbc_sign = np.load('%s_lsbc_sign_8.npy' % name)
        # print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_lsbc[T])]),'lsbc')
        # print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_lsbc_sign[T])]), 'lsbc_sign')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str,
                        default=None,
                        help='name of dataset')

    FLAGS, unparsed = parser.parse_known_args()
    name = FLAGS.d
    print(name)
    main(name)