import numpy as np
import argparse
def main(name):
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

    non_zero_2 = np.load('%s_non0_random_init_8.npy' % name)
    # non_zero_3 = np.load('%s_non0_cos_similarity.npy' % name)


    # for T in range(6):
    acc_Fastfood = np.load('%s_Fastfood_8.npy' % name)
    acc_RKS = np.load('%s_RKS_8.npy' % name)
    #
    print(acc_Fastfood,acc_RKS)
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_Fastfood)]),'fastfood' )
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_RKS)]), 'RKS')
    acc_scbd = np.load('%s_sign_8.npy' % name)
    acc_scbd_sign = np.load('%s_sign_sign_8.npy' % name)
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_scbd)]),'sign_linear')
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_scbd_sign)]), 'sign_sign')
    acc_our = np.load('%s_random_init_8.npy' % name)
    print(acc_scbd)
    acc_our_linear = np.load('%s_random_init_linear_8.npy' % name)
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_our)] ),'our proposed')
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_our_linear)]), 'our linear')
    print(acc_our, acc_our_linear,non_zero_2)
    acc_lsbc = np.load('%s_lsbc_8.npy' % name)
    acc_lsbc_sign = np.load('%s_lsbc_sign_8.npy' % name)
    print(acc_lsbc_sign)
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_lsbc)]),'lsbc')
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in [np.max(acc_lsbc_sign)]), 'lsbc_sign')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str,
                        default=None,
                        help='name of dataset')

    FLAGS, unparsed = parser.parse_known_args()
    name = FLAGS.d
    print(name)
    main(name)