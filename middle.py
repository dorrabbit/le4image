from numpy.random import *
import numpy as np
from dropout import Dropout
import math

class Middle:
    def middle(xlist, prenum, nownum, w, b, batchnum, itbool, srclass, normclass):
        #random
        #seed(seednum)
        #wran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, prenum))
        #seed(seednum)
        #bran = normal(loc = 0, scale = 1/math.sqrt(prenum) , size = (nownum, 1))

        xlist = xlist.reshape(-1,batchnum)
        summid = np.dot(w, xlist) + b

        #print(summid.shape)
        (sum_norm, running_mean, running_var) = normclass.forward(summid, itbool)
        
        sig = srclass.forward(sum_norm)
        #sig.shape=[55,100]
        
        return (sig, running_mean, running_var)
