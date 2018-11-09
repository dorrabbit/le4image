import numpy

from inputer import Inputer
from sigmoid import Sigmoid
from relu import Relu
from dropout import Dropout
dropclass = Dropout()
from batchnorm import Batchnorm
from adam import Adam

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

[w_one, w_two, b_one, b_two, gamma_mid, beta_mid, gamma_out, beta_out,\
 running_mean_mid, running_var_mid, running_mean_out, running_var_out] = numpy.load(npyfile)

normclass_mid = Batchnorm(gamma_mid, beta_mid, 0.9, running_mean_mid, running_var_mid)
normclass_out = Batchnorm(gamma_out, beta_out, 0.9, running_mean_out, running_var_out)
loopnum = 10000
truenum = 0
for i in range(loopnum):
    innum = numpy.random.choice(trainsize, replace=True)
    
    xlist = (dlist[innum] / 255).T
    ylist = anslist[innum]
    y_onehot = numpy.identity(10)[ylist]
    
    outrslt = threeclass.three_nn(xlist, y_onehot, w_one, b_one, w_two, b_two, \
                                  1, True, srclass, dropclass, normclass_mid, normclass_out)[1]
    
    rslt = numpy.argmax(outrslt)

    accbool = ylist == rslt

    if accbool:
        truenum = truenum + 1
    #else:
    #    print(innum, ylist, rslt)
        
print("accuracy is", (truenum / loopnum) * 100, "%.")

    
