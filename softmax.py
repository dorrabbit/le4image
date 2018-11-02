import numpy as np
import numpy.matlib
class Softmax:
    def softmax(midrslt, batchnum):
        #print(midrslt.shape)
        maxa = (midrslt.max(axis=0)).reshape(1, batchnum)
        suma = ((np.exp(midrslt - maxa)).sum(axis=0)).reshape(1, batchnum)
        ysec = np.exp(midrslt - maxa) / np.matlib.repmat(suma, 10, 1)
        return ysec
