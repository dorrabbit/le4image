import numpy as np
from numpy.random import *

class Dropout:
    def __init__(self, rho=0):
        self.rho = rho
        self.mask = None
        
    def forward(self, sig, itbool):
        if itbool:
            return sig * (1 - self.rho)
        else:
            self.mask = np.random.rand(*sig.shape) < self.rho
            sig[self.mask] = 0
            return sig

    def backward(self, dout):
        dout[self.mask] = 0
        return dout
