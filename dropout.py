import numpy as np
from numpy.random import *

class Dropout:
    def dropout(self, sig, rho, midnum):
        #sig.shape=[55,100]
        sigt = sig.T
        for i in range(99):
            self.rannumlist = np.random.choice(midnum, int(midnum * rho), replace=False)
            sigt[i][self.rannumlist]=0
        droprslt = sigt.T
        return droprslt

    def d_dropout(self):
        return self.rannumlist
