import numpy as np

def likelihood(x):
    '''
    LIKELIHOOD Different Class Feature Liklihood 
    INPUT:  x, features of different class, C-By-N numpy array
            C is the number of classes, N is the number of different feature

    OUTPUT: l,  likelihood of each feature(from smallest feature to biggest feature) given by each class, C-By-N numpy array
    '''

    C, N = x.shape
    l = np.zeros((C, N))


    # begin answer
    # print(x)
    # sum of each class
    total = np.sum(x, axis = 1)
    # print(total)

    for i in range(C):
        for j in range(N):
            l[i,j] = x[i,j] / total[i]

    

    # end answer
    return l