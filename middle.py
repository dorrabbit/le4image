from numpy.random import *
import numpy as np
from dropout import Dropout
import math

class Middle:
    def middle(self, xlist, prenum, nownum, w, b, batchnum, srclass):
        #random
        #seed(seednum)
        #wran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, prenum))
        #seed(seednum)
        #bran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, 1))

        xlist = xlist.reshape(prenum,batchnum)
        summid = np.dot(w, xlist) + b

        #print(summid.shape)
        sig = srclass.forward(summid)
        #sig.shape=[55,100]
        
        return sig
