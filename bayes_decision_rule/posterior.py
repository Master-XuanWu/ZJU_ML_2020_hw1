import numpy as np
from likelihood import likelihood

def posterior(x):
    '''
    POSTERIOR Two Class Posterior Using Bayes Formula
    INPUT:  x, features of different class, C-By-N vector
            C is the number of classes, N is the number of different feature
    OUTPUT: p,  posterior of each class given by each feature, C-By-N matrix
    '''

    C, N = x.shape
    l = likelihood(x)
    # total of occurences of features
    total = np.sum(x)
    p = np.zeros((C, N))


    # begin answer
    total_per_feature = np.sum(x, axis = 0)
    total_per_class = np.sum(x, axis = 1)
    prior = np.zeros(C)
    for i in range(C):
            prior[i] = total_per_class[i] / total
    prob_per_feature = np.zeros(N)
    for i in range(N):
        prob_per_feature[i] = total_per_feature[i] / total

    for i in range(C):
        for j in range(N):
            p[i,j] = l[i,j] * prior[i] / prob_per_feature[j]
    # end answer
    
    return p
