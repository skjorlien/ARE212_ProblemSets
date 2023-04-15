import numpy as np
from scipy.stats import t

class Two_SLS:
    def __init__(self, y, X, Z):
        self.y = y
        self.X = X
        self.Z = Z
        self.beta_tsls = self._compute_beta_hat()
        self.df = self.X.shape[0]-self.X.shape[1]

    def _compute_beta_hat(self):
        xhat = self.Z@np.linalg.inv(self.Z.T@self.Z)@self.Z.T@self.X
        beta_tsls = np.linalg.solve(xhat.T@xhat, xhat.T@self.y)
        return beta_tsls
    
    def variance(self):
        return self.residuals().var()*np.linalg.inv(self.X.T@self.X)

    def residuals(self):
        return self.y - self.beta_tsls*self.X
    
    def se(self):
        variances = np.diag(self.variance())
        se = np.sqrt(variances)
        return se

    def ttest(self, testval = 0, slice = None):
        tstat = (self.beta_tsls - testval)/self.se()
        tstat.flatten()[0]
        return tstat.flatten()[0]
        
    def ci(self, CI): # not quite working ... fml maybe should just use built in package I think you can.
        ci = t.interval(CI,df = self.df, loc=self.beta_tsls, scale = self.se()) 

        return (ci[0][0][0], ci[1][0][0])
    
    def p_val(self):
        tstat = self.ttest()
        p = 2*(1 - t.cdf(abs(tstat), self.df))
        return p
