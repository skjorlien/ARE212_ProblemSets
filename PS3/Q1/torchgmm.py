import numpy as np
import torch as torch

################### 
# This notebook aims to build a script that can run GMM
# estimation routines. It is a PyTorch implementation of the gmm.py file
# provided by Ethan Ligon. It isn't as robust to inputs as I didn't want to
# write up try/except and assert commands which are best practice.
###################

def linearModel_moments(b, y, X, Z):
    """
    Moments for following model:
    
    Linear model: y = X*b + u where E[u|Z] =0.
    """
    n, k = Z.shape # Requires Z to be of matrix dim.
    res =  Z * (y - X @ b) 
    res = res.reshape(n,k) # TODO: check if this is needed.
    return res


def gj(b, data, f_m=linearModel_moments):
    """
    This is a wrapper function for general moment functions. 
    """
    return f_m(b, *data)

def gN(b, data, f_m=linearModel_moments):
    """
    This function gets the average moment
    """
    g_j = gj(b, data, f_m=f_m)
    N   = g_j.shape[0]
    return g_j.mean(0, keepdim=True), N

def Omegahat(b, data, f_m=linearModel_moments):
    """
    Function to get Var(u).
    """
    res = gj(b, data, f_m=f_m)
    res_c = res - res.mean(0, keepdim=True) # center

    return (res_c.T @ res_c)/res.shape[0]

def J(b, W, data, f_m=linearModel_moments):
    """
    Function to get objective value.
    """
    g_N, N = gN(b, data, f_m=f_m)
    return N*(g_N @ W @ g_N.T).squeeze()

def solve_kstep_GMM(steps, beta_init, data, n_m, device, t_type, f_m=linearModel_moments, final_verbose=True):
    # Get beta parameter as a pytorch parameter
    beta = torch.as_tensor(beta_init, device=device, dtype=t_type).requires_grad_(True)
    # Define first step weights: identity matrix
    weights = torch.eye(n_m, device=device, dtype=t_type)
    # Define optimization algorithm
    opt = torch.optim.LBFGS(
        [beta],
        max_iter=200,
        line_search_fn='strong_wolfe', tolerance_grad=1e-10)
    # Get objective function conforming to a pytorch solver
    def obj(weighting_matrix=weights,verbose=False):
        opt.zero_grad()
        res = J(beta, weights, data, f_m=f_m)
        res.backward()
        if verbose:
            print('GMM-{0} loss: {1:.6f}'.format(step+1, res.item()))
        return res

    # Execute k-th step of GMM.
    for step in range(steps):
        opt.step(obj)
        weights = torch.inverse(Omegahat(beta.detach().clone(),data, f_m=f_m)) # Updating weights
        if final_verbose:
            print('GMM-{0} loss: {1:.6f}'.format(step+1, obj().item()))
    
    # return parameters and weights
    return beta.detach().clone(), weights.detach().clone()