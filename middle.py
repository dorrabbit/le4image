from numpy.random import *
import numpy as np
from dropout import Dropout
import math

class Middle:
    def middle(self, xlist, prenum, nownum, w, b, batchnum, itbool, srbool, srclass):
        #random
        #seed(seednum)
        #wran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, prenum))
        #seed(seednum)
        #bran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, 1))

        xlist = xlist.reshape(prenum,batchnum)
        summid = np.dot(w, xlist) + b

        #print(summid.shape)
        if srbool:
            sig = srclass.forward(summid)
        else:
            sig = Relu.relu(summid)
        #sig.shape=[55,100]
        
        #dropout
        #rho = 0.5
        #if itbool:
        #    midrslt = sig * (1-rho)
        #else:
        #    dropclass = Dropout()
        #    midrslt = dropclass.dropout(sig, rho, nownum)
        midrslt=sig
        return midrslt
