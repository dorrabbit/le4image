import numpy as np
import numpy.matlib
class Softmax:
    def softmax(self, midrslt):
        maxa = (midrslt.max(axis=0)).reshape(1, 100)
        suma = ((np.exp(midrslt - maxa)).sum(axis=0)).reshape(1, 100)
        ysec = np.exp(midrslt - maxa) / np.matlib.repmat(suma, 10, 1)
        return ysec
