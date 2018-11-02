import numpy

from inputer import Inputer
from sigmoid import Sigmoid
from relu import Relu
from dropout import Dropout
dropclass = Dropout()
from batchnorm import Batchnorm

inclass = Inputer()
dlist = inclass.inputer(True)[0]
#print(dlist)
anslist = inclass.inputer(True)[1]
#print(ylist)
trainsize = dlist.shape[0] #N

sig_relu = input("sigmoid or relu? s/r > ")
####
if sig_relu == "s":
    srclass = Sigmoid()
    npyfile = 'wb_learn_s.npy'
elif sig_relu == "r":
    srclass = Relu()
    npyfile = 'wb_learn_r.npy'
####

from three_nn import Three_nn
threeclass = Three_nn()

midnum = 55 #middle_node_of_number

[w_one, w_two, b_one, b_two, gamma, beta, running_mean, running_var] = numpy.load(npyfile)

normclass = Batchnorm(gamma, beta, 0.9, running_mean, running_var)
loopnum = 10000
truenum = 0
for i in range(loopnum):
    innum = numpy.random.choice(trainsize, replace=True)
    
    xlist = (dlist[innum] / 255).T
    ylist = anslist[innum]
    y_onehot = numpy.identity(10)[ylist]
    
    outrslt = threeclass.three_nn(xlist, y_onehot, w_one, b_one, w_two, b_two, 1, True, srclass, dropclass, normclass)[1]
    
    rslt = numpy.argmax(outrslt)

    accbool = ylist == rslt

    if accbool:
        truenum = truenum + 1
    #else:
    #    print(innum, ylist, rslt)
        
print("accuracy is", (truenum / loopnum) * 100, "%.")

    
