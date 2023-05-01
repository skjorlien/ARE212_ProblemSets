"""
Functions for problem 3
"""
import numpy as np
import scipy.stats as iid
from scipy.optimize import minimize

##### data generating processess 
# dgp for normal distrubted x
def dgp_norm(N, mu, sigma2):
    
    x = iid.norm.rvs(mu, sigma2, size = N)
    
    return x

# create dgp for non-normal distributed x (uniform)
def dgp_unif(N, minn, maxx):
    
    x = iid.uniform.rvs(minn, maxx-minn, size = N) 
    #gives uniform (loc, loc+scale) so need to subtract off loc
    
    return x


##### GMM functions
# double factorial (need in gi)
def dfact(n):
     if n < 2:
         return 1
     else:
         return n * dfact(n-2)


# gi for different values of k    
def gi(param, K, x):
    '''
    moment restrictions for different k
    E(x) = mu
    E(x-mu)^k = 0 k odd
    E(x-mu)^k = sigma^k (k-1)!!
    '''
    
    mu = param[0]
    sigma = param[1]
    
    moment_fun = lambda k: (x-mu)**k - (sigma**k)*(dfact(k-1)) if k%2 == 0 else (x-mu)**k
    
    moments = np.array([moment_fun(k+1) for k in range(K)])
    
    return moments.T #return column vector

def gN(param, K, x):
    
    mu = param[0]
    sigma = param[1]
    
    # get individual moments
    e = gi(param, K, x)
    # take mean
    gN = e.mean(axis=0).reshape((K,1))
    return gN

def invOmega(param, K, x):
    
    e = gi(param, K, x)
    # recenter
    e = e - e.mean(axis=0)
    N = e.shape[0]
    var = e.T@e/N
    return np.linalg.inv(var)

def J(param, K, W, x):

    m = gN(param, K, x) # Sample moments
    
    N = x.shape[0]

    return N*m.T@W@m # Scale by sample size

# two step gmm function that returns estimates and J
def two_step_gmm(K, x):
    
    # Implement gmm with sub-optimal weighting matrix
    Omega_guess = np.eye(K)
    
    minimizer = minimize(lambda param: J(param, K, Omega_guess, x), x0 = [0,0], method = 'Nelder-Mead')
    param_hat = minimizer.x
    
    # Update Omega
    invOmega_hat = invOmega(param_hat, K, x)
    
    # re-run with updated weighting matrix
    minimizer2 = minimize(lambda param: J(param, K, invOmega_hat, x), param_hat, method = 'Nelder-Mead')
    param_hat2 = minimizer2.x
    
    # get J with new params
    J_val = J(param_hat2, K, invOmega_hat, x)
    
    # return both the coefficients and J
    return param_hat2, J_val

###### MLE function
def MLE_norm(param, data):
    mu = param[0]
    sigma = param[1]
    
    # Calculate the log-likelihood for normal distribution
    LL = np.sum(iid.norm.logpdf(data, mu, sigma))
    
    # Calculate the negative log-likelihood
    return -1*LL #minimize 
