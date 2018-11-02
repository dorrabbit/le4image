from numpy.random import *
import numpy as np
from softmax import Softmax
import math

class Outnode:
    def outnode(self, ylist, midnum, w, b, batchnum, itbool, normclass):
        #random
        #seed(seednum)
        #wran = normal(loc = 0, scale = 1/math.sqrt(midnum), size = (10, midnum))
        #seed(seednum)
        #bran = normal(loc = 0, scale = 1/math.sqrt(midnum), size = (10, 1))

        summid = np.dot(w, ylist) + b

        (sum_norm, running_mean, running_var) = normclass.forward(summid, itbool)
    
        soft = Softmax.softmax(sum_norm, batchnum)
        return (soft, running_mean, running_var)
