from scipy.stats import distributions as iid
from scipy.stats import multivariate_normal
import numpy as np 
import matplotlib.pyplot as plt
from classes import Two_SLS


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


def hansen(x, y, Z, b0):
    ybar = y - b0*x
    gammahat = np.linalg.solve(Z.T@Z, Z.T@ybar)
    e = ybar - Z@gammahat
    df = Z.shape[0] - Z.shape[1]
    V = compute_variance(e, Z, df)
    # define f statistic 
    df1 = gammahat.size ## number of restrictions (setting all gamma = 0)
    df2 = df ## N - number of parameters 
    Q = Z@(V/Z.shape[0])@Z.T
    fstat = (Z@gammahat).T@np.linalg.pinv(Q)@(Z@gammahat)
    p = 2*(1 - iid.f.cdf(fstat, df1, df2))
    return p


n = 1000
pi = np.linspace(0.001,1,9)
fig, ax = plt.subplots(3,3)
ax = ax.reshape(-1)
for i, coef in enumerate(pi): 
    b = np.empty(0)
    se = np.empty(0)
    for j in range(1000):
        x, y, Z = dgp(n, β, π)
        model = Two_SLS(y, x, Z)
        b = np.append(b, model.beta_tsls)
        # se = np.append(se, model.se())

    
    monte_carlo_5 = np.percentile(b, 5)
    monte_carlo_95 = np.percentile(b, 95)

    ax[i].hist(b)
    ax[i].set_title(f"Pi = {np.round(coef, 2)}")
    ax[i].axvline(x=1, color='r', label='True beta')
    ax[i].axvline(x = monte_carlo_5, color = 'b')
    ax[i].axvline(x = monte_carlo_95, color = 'b')
fig.tight_layout()