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

