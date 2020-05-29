import numpy as np
import argparse
def main(name):
    acc_1 = []
    acc_2 = []
    acc_3 = []
    acc_fastfood = []
    acc_rks = []
    acc_Fastfood = np.load('%s_Fastfood.npy' % name )
    acc_RKS = np.load('%s_RKS.npy' % name )
    if 1:
        acc_binary_1 = np.load('%s_sign.npy' % name )

        acc_binary_2 = np.load('%s_0_init.npy' % name )
        acc_binary_3 = np.load('%s_lsbc.npy' % name )
        for T in range(6):
            acc_1.append(np.max(np.mean(acc_binary_1[T].reshape(3,-1),axis=0)))
            acc_2.append(np.max(np.mean(acc_binary_2[T].reshape(3, -1), axis=0)))
            acc_3.append(np.max(np.mean(acc_binary_3[T].reshape(3, -1), axis=0 )))
        non_zero_1 = np.load('%s_non0_sign.npy' % name)

        non_zero_2 = np.load('%s_non0_0_init.npy' % name)
        # non_zero_3 = np.load('%s_non0_cos_similarity.npy' % name)

        # print(acc_binary_3)
        for T in range(6):

            acc_rks.append(np.max(acc_RKS[T]))
            acc_fastfood.append(np.max(acc_Fastfood[T]))
    print(acc_1,acc_2,acc_3)
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in acc_1),'sign')
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in acc_2),'our proposed')
    print(" & ".join(str(np.around(i * 100, decimals=2)) for i in acc_3),'lsbc')
    print(" & ".join(str(np.around(i*100,decimals=2)) for i in acc_fastfood))
    print(" & ".join(str(np.around(i*100,decimals=2)) for i in acc_rks))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str,
                        default=None,
                        help='name of dataset')

    FLAGS, unparsed = parser.parse_known_args()
    name = FLAGS.d
    print(name)
    main(name)