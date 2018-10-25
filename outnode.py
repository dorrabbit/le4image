from numpy.random import *
import numpy as np
from softmax import Softmax
import math

class Outnode:
    def outnode(self, ylist, midnum, w, b, batchnum):
        #random
        #seed(seednum)
        #wran = normal(loc = 0, scale = 1/math.sqrt(midnum), size = (10, midnum))
        #seed(seednum)
        #bran = normal(loc = 0, scale = 1/math.sqrt(midnum), size = (10, 1))

        summid = np.dot(w, ylist) + b

        softclass = Softmax()
        soft = softclass.softmax(summid, batchnum)
        return soft
