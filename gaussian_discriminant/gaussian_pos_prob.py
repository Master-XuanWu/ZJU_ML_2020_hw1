import numpy as np
import math

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1] # N data points
    K = Phi.shape[0] # K classes
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    for n in range(N):
        # intialization
        prob_x = 0
        x = X[:, n]
        for k in range(K):
            # calc likelihood * prior using x, Mu[:, k], Sigma[:, :, k], Phi[k]
            p[n, k] = math.exp(-0.5 * \
                np.dot(np.transpose(x-Mu[:, k]), np.dot(np.linalg.inv(Sigma[:, :, k]), x-Mu[:, k]))) / \
                math.sqrt(np.linalg.det(Sigma[:, :, k])) * Phi[k]
            # add likelihood into P(x)
            prob_x += p[n, k]
        # finally, compute each posterior  
        p[n] /= prob_x

    # end answer
    
    return p
    
