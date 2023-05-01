################### 
# This notebook aims to build a script for different data
# generating processes estimation routines.
###################
import numpy as np
import torch
from scipy.stats import distributions as iid
from scipy.stats import multivariate_normal
from numpy.linalg import inv

class dgp():
    """Convolve two discrete random variables s and t.
    So we want to return the properties of k = s + t"""
    def __init__(self, seed, N, t_type, device):
        # Constructor
        self.seed = seed
        self.N    = N
        self.t_type = t_type
        self.device = device

    def part_a(self, mu=1, s2=2):
        np.random.seed(self.seed)
        y = iid.norm().rvs(size=(self.N,1))*s2**.5 + mu
        y = torch.tensor(y, dtype=self.t_type, device=self.device)
        return (y,)
    
    def part_b_c(self, beta=[2, 4, 3], s2=1):
        beta = np.array(beta).reshape(-1,1)
        k    = beta.shape[0]
        np.random.seed(self.seed)
        u = iid.norm().rvs(size=(self.N,1))*s2**.5
        X = iid.norm().rvs(size=(self.N, k))
        X[:,0] = 1 # first column is a constant
        y = X @ beta + u
        y = torch.tensor(y, dtype=self.t_type, device=self.device)
        X = torch.tensor(X, dtype=self.t_type, device=self.device)
        return y, X, X
    
    def part_d(self, beta=[2, 4], s2=1):
        beta = np.array(beta).reshape(-1,1)
        if beta.shape[0]>2:
            print("This dgp code wasn't written for more than two parameters.")
        k    = beta.shape[0]
        np.random.seed(self.seed)
        u = iid.norm().rvs(size=(self.N,1))
        X = iid.norm().rvs(size=(self.N, k))
        X[:,0] = 1 # first column is a constant
        u = u * np.exp(X[:,[1]]*s2)**.5 # Creating heteroskedastic data.
        y = X @ beta + u
        y = torch.tensor(y, dtype=self.t_type, device=self.device)
        X = torch.tensor(X, dtype=self.t_type, device=self.device)
        
        return y, X, X
    
    def part_e(self, beta=[2,4], l=2):
        beta = np.array(beta).reshape(-1,1)

        k    = beta.shape[0] -1 # non-constant dim of X
        if l < k+1:
            print("l set to too low. Setting it to the dim of X")
            l = k+1

        np.random.seed(self.seed)
        mu = np.random.rand(k+l+1,1) # random draws of the mean values
        A = np.random.rand(k+l+1,k+l+1)
        mu[k+l,:] = 0 # average of error term is 0
        covXZu = A.T @ A  + np.eye(k+l+1)*2  # make it psd
        covXZu[k+l,k:k+l]=0 # setting cov(Z, u)=0
        covXZu[k:k+l,k+l]=0 # setting cov(Z, u)=0
        XZu = multivariate_normal(mu.flatten(), covXZu).rvs(size=self.N)

        X   = np.zeros((self.N, k+1))
        X[:,0] = 1 # first column is a constant
        X[:,1:] = XZu[:,0:k]
        Z   = np.zeros((self.N, l))
        Z = XZu[:,k:(k+l)]
        u = XZu[:,[k+l]]

        y = X @ beta + u
        y = torch.tensor(y, dtype=self.t_type, device=self.device)
        X = torch.tensor(X, dtype=self.t_type, device=self.device)
        Z = torch.tensor(Z, dtype=self.t_type, device=self.device)
        
        return (y, X, Z), covXZu

    def part_f(self, beta=[2,4], l=2, f=lambda x: x**2):
        beta = np.array(beta).reshape(-1,1)

        k    = beta.shape[0] -1 # non-constant dim of X
        if l < k+1:
            print("l set to too low. Setting it to the dim of X")
            l = k+1

        np.random.seed(self.seed)
        mu = np.random.rand(k+l+1,1) # random draws of the mean values
        A = np.random.rand(k+l+1,k+l+1)
        mu[k+l,:] = 0 # average of error term is 0
        covXZu = A.T @ A  + np.eye(k+l+1)*2  # make it psd
        covXZu[k+l,k:k+l]=0 # setting cov(Z, u)=0
        covXZu[k:k+l,k+l]=0 # setting cov(Z, u)=0
        XZu = multivariate_normal(mu.flatten(), covXZu).rvs(size=self.N)

        X   = np.zeros((self.N, k+1))
        X[:,0] = 1 # first column is a constant
        X[:,1:] = XZu[:,0:k]
        Z   = np.zeros((self.N, l))
        Z = XZu[:,k:(k+l)]
        u = XZu[:,[k+l]]

        X = torch.tensor(X, dtype=self.t_type, device=self.device)
        Z = torch.tensor(Z, dtype=self.t_type, device=self.device)
        
        y = f(X @ beta) + u
        
        return (y, X, Z, f), covXZu
    
    def part_g(self, beta=[2,4], l=2, f=lambda x, b: x**2 @ b**2):
        beta = np.array(beta).reshape(-1,1)

        k    = beta.shape[0] -1 # non-constant dim of X
        if l < k+1:
            print("l set to too low. Setting it to the dim of X")
            l = k+1

        np.random.seed(self.seed)
        mu = np.random.rand(k+l+1,1) # random draws of the mean values
        A = np.random.rand(k+l+1,k+l+1)
        mu[k+l,:] = 0 # average of error term is 0
        covXZu = A.T @ A  + np.eye(k+l+1)*2  # make it psd
        covXZu[k+l,k:k+l]=0 # setting cov(Z, u)=0
        covXZu[k:k+l,k+l]=0 # setting cov(Z, u)=0
        XZu = multivariate_normal(mu.flatten(), covXZu).rvs(size=self.N)

        X   = np.zeros((self.N, k+1))
        X[:,0] = 1 # first column is a constant
        X[:,1:] = XZu[:,0:k]
        Z   = np.zeros((self.N, l))
        Z = XZu[:,k:(k+l)]
        u = XZu[:,[k+l]]

        X = torch.tensor(X, dtype=self.t_type, device=self.device)
        Z = torch.tensor(Z, dtype=self.t_type, device=self.device)
        
        y = f(X, beta) + u
        
        return (y, X, Z, f), covXZu
    
    def part_h(self, alpha=1., gamma=3.):
        np.random.seed(self.seed)
        v = iid.uniform().rvs(size=(self.N, 1))*2-1
        u = v**gamma
        Z = v**(gamma-1)
        y_gamma = u + alpha
        y = y_gamma**(1/gamma)

        y = torch.tensor(y, dtype=self.t_type, device=self.device)
        Z = torch.tensor(Z, dtype=self.t_type, device=self.device)

        return (y, Z)

def moment_functions(part):
    if part=="a":
        f_m = lambda b, y: torch.concat([y-b[0], (y - b[0])**2-b[1],(y - b[0])**3],dim=1)
    if part=="b":
        def f_m(b, y, X, Z):
            n, k = Z.shape # Requires Z to be of matrix dim.
            res = Z * (y - X @ b)
            res = res.reshape(n, k)
            return res
    if part=="c":
        def f_m(b, y, X, Z):
            beta = b[:-1]
            sigma = b[-1:]
            n, k = Z.shape # Requires Z to be of matrix dim.
            res = torch.concat([Z * (y - X @ beta), (y-X@beta)**2 - sigma],axis=1)
            res = res.reshape(n, k+1)
            return res
    if part=="d":
        def f_m(b, y, X, Z):
            beta = b[:-1]
            sigma = b[-1:]
            n, k = Z.shape # Requires Z to be of matrix dim.
            res = torch.concat([Z * (y - X @ beta), (y-X@beta)**2 - torch.exp(X[:,[1]]*sigma)],axis=1)
            res = res.reshape(n, k+1)
            return res
    if part=="e":
        def f_m(b, y, X, Z):
            n, k = Z.shape # Requires Z to be of matrix dim.
            res = torch.concat([Z * (y - X @ b), y-X@b],axis=1)
            res = res.reshape(n, k+1)
            return res
    if part=="f":
        def f_m(b, y, X, Z, g):
            n, k = Z.shape # Requires Z to be of matrix dim.
            res = torch.concat([Z * (y - g(X @ b)), y-g(X@b)],axis=1)
            res = res.reshape(n, k+1)
            return res
    if part=="g":
        def f_m(b, y, X, Z, g):
            n, k = Z.shape # Requires Z to be of matrix dim.
            res = torch.concat([Z * (y - g(X, b)), y-g(X,b)],axis=1)
            res = res.reshape(n, k+1)
            return res
        
    if part=="h":
        def f_m(b, y, Z):
            n, k = Z.shape # Requires Z to be of matrix dim.
            res = torch.concat([Z * (y**b[0] - b[1]), y**b[0] - b[1]],axis=1)
            res = res.reshape(n, k+1)
            return res
    return f_m