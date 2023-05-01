from scipy.stats import distributions as iid
from scipy.stats import multivariate_normal
import numpy as np 
import matplotlib.pyplot as plt
from classes import Two_SLS
from statsmodels.regression.linear_model import OLS
from scipy.optimize import minimize

def dgp(n, β, π):
    ## Generate errors 
    u = iid.norm().rvs(size = (n, 1))
    v = iid.norm().rvs(size= (n, 1))

    ## the number of instruments to generate = number of elements in pi
    if type(π) == np.ndarray:
        if π.ndim == 1: 
            π = π.reshape(-1, 1)
    else: 
        π = np.array([π]).reshape(-1,1)
    l = π.size

    # create x, y, Z
    Z = multivariate_normal(np.zeros(l), np.eye(l)).rvs(size=n)
    if Z.ndim == 1: 
        Z = Z.reshape(-1, 1)
    x = Z@π + v
    y = β*x + u 
    return (y, x, Z) 


def chern_hansen(y,X,Z,b0):
    lhs = y-b0*X
    model = OLS(lhs, Z)
    results = model.fit()
    
    A = np.identity(len(results.params))
    fresults = results.f_test(A)
    fval = fresults.fvalue
    pval = fresults.pvalue
    
    return -pval 

def do_two_sls(n, beta, pi):
    x, y, Z = dgp(n, beta, pi)
    model = Two_SLS(y, x, Z)
    return model.beta_tsls.flatten()[0], model.ttest(), model.p_val(), *model.ci(0.95)

def do_chern_hansen(n, beta, pi):
    x, y, Z = dgp(n, beta, pi)
    beta_hat = minimize(lambda b: chern_hansen(y,x,Z,b), x0=beta, method = 'Nelder-Mead').x
    return beta_hat

def calc_coverage(est, crit = 1.96):
    ## Get coverage of Chern-Hansen
    ci_lower = est.mean() - crit*np.sqrt(est.var())
    ci_upper = est.mean() + crit*np.sqrt(est.var())
    coverage = ci_upper - ci_lower 
    return coverage

def mega_dgp(n,k, β, π):
    u = iid.norm().rvs(size = (n*k, 1))
    v = iid.norm().rvs(size= (n*k, 1))

    ## the number of instruments to generate = number of elements in pi
    if type(π) == np.ndarray:
        if π.ndim == 1: 
            π = π.reshape(-1, 1)
    else: 
        π = np.array([π]).reshape(-1,1)
    l = π.size

    # create x, y, Z
    Z = multivariate_normal(np.zeros(l)+2, np.eye(l)).rvs(size=n*k)
    if Z.ndim == 1: 
        Z = Z.reshape(-1, 1)
        
    x = Z@π + v
    y = β*x + u 
    
    y = y.reshape((k, n))
    x = x.reshape((k, n))
    Z = Z.reshape((k, n, -1))
    return (y, x, Z) 

def dirty_TSLS(y, X, Z):
    # Quick and dirty TSLS implementation
    xhat      = Z @ np.linalg.solve(Z.T@Z, Z.T@X)
    beta_tsls = np.linalg.solve(xhat.T@xhat, xhat.T@y)
    res       = y - beta_tsls*X
    se       = np.sqrt(res.var()/(X.T@X))
    return np.array([beta_tsls, se]).flatten()

def run_many_instruments_monte_carlo(n_list, beta, N, L, K):
    coverage = np.zeros((N,len(n_list)))
    rmse     = np.zeros((N,len(n_list)))
    mean_se = np.zeros((N,len(n_list)))

    for (ii, n) in enumerate(n_list):
        for l in L:
            i = l-1
            pi = np.array([(.5)**(j) for j in range(l)])
            y_batch, x_batch, Z_batch = mega_dgp(n, K, beta, pi)
            b_se = [dirty_TSLS(y_batch[[k],:].T, x_batch[[k],:].T, Z_batch[k,:,:]) for k in range(K)]
            b_se = np.array(b_se)
            b     = b_se[:,0]
            se    = b_se[:,1]
            rmse[i, ii]       = (((b-beta)**2).mean())**.5
            mean_se[i, ii]    = se.mean()
            coverage[i, ii]   = calc_coverage(b)

    return rmse, mean_se, coverage