from numpy.random import *
import numpy as np
from softmax import Softmax
import math

class Outnode:
    def outnode(self, ylist, midnum, seednum):
        #random
        seed(seednum)
        wran = normal(loc = 0, scale = 1/math.sqrt(midnum), size = (10, midnum))
        seed(seednum)
        bran = normal(loc = 0, scale = 1/math.sqrt(midnum), size = (10, 100))
        summid = (np.dot(wran, ylist) + bran).T

        #print(summid)

        softclass = Softmax()
        soft = softclass.softmax(summid)
        return soft
