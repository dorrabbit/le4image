import numpy as np
import numpy.matlib
class Softmax:
    def softmax(self, midrslt):
        maxa = (midrslt.max(axis=1)).reshape(100, 1)
        suma = ((np.exp(midrslt - maxa)).sum(axis=1)).reshape(100, 1)
        ysec = np.exp(midrslt - maxa) / np.matlib.repmat(suma, 1, 10)
        return ysec
