from scipy.stats import distributions as iid
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

class ConvolveDiscrete(iid.rv_discrete):
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        super().__init__()

    def _cdf(self, z):
        F = 0 
        for val, prob in zip(self.x.xk, self.x.pk):
            F += self.y.cdf(z - val)* prob
        return F

    def _pmf(self, z):
        f = 0 
        for val, prob in zip(self.x.xk, self.x.pk):
            f += self.y.pmf(z - val)* prob
        return f

class ConvolveContinuous(iid.rv_continuous):
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        super().__init__()

    def _cdf(self, z):
        return integrate.quad(lambda x: self.y.cdf(z - x) * self.x.pdf(x), -np.inf, np.inf)

    def _pdf(self, z):
        return integrate.quad(lambda x: self.y.pdf(z - x) * self.x.pdf(x), -np.inf, np.inf)


if __name__ == "__main__":
    # create discrete random variables
    Omega = (1, 2, 3, 4, 5, 6)
    Pr = tuple([1/6]*6)
    Dice1 = iid.rv_discrete(values=(Omega, Pr))
    Dice2 = iid.rv_discrete(values=(Omega, Pr))
    
    Omega = (0, 1)
    Pr = (.5, .5)
    Coin = iid.rv_discrete(values=(Omega, Pr))

    # create continuous random variables
    Normal1 = iid.norm()
    Normal2 = iid.norm()
    
    z = ConvolveContinuous(Normal1, Normal2) 

    # Plot convolved against the standard normal
    X = np.linspace(-4, 4, 100).tolist()
    Y = [z.pdf(x) for x in X]
    stdY = [Normal1.pdf(x) for x in X]
    plt.plot(X, Y)
    plt.plot(X, stdY)
    plt.show()