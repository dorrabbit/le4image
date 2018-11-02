import numpy
from inputer import Inputer
inclass = Inputer()
from three_nn import Three_nn
threeclass = Three_nn()
from numpy.random import *
from back import Back
backclass = Back()
from hyojun_in import Hyojun_in
hyoinclass = Hyojun_in()
from sigmoid import Sigmoid
from relu import Relu
#
from batchnorm import Batchnorm
batchmoment = 0.9
#
from dropout import Dropout
dropclass = Dropout()
#
import sys

iden_train = input("identification or training? i/t > ")
####
if iden_train == "i":
    itbool = True
elif iden_train == "t":
    itbool = False
####

sig_relu = input("sigmoid or relu? s/r > ")
sys.stdout.write("Now loading...")
sys.stdout.flush()
####
if sig_relu == "s":
    npyfile = 'wb_learn_s.npy'
    srclass = Sigmoid()
elif sig_relu == "r":
    npyfile = 'wb_learn_r.npy'
    srclass = Relu()
####

dlist = inclass.inputer(itbool)[0]
#print(dlist)
anslist = inclass.inputer(itbool)[1]
#print(ylist)
trainsize = dlist.shape[0] #N

midnum = 55 #middle_node_of_number

[w_one, w_two, b_one, b_two, gamma, beta, running_mean, running_var] = numpy.load(npyfile)
#w_one = numpy.load(npyfile)[0]
#w_two = numpy.load(npyfile)[1]
#b_one = numpy.load(npyfile)[2]
#b_two = numpy.load(npyfile)[3]
#gamma = num

if not itbool:
    load_yn = input("\r" + "use saved weight? y/n > ")
    if load_yn == "n":
        #initialize
        w_one = normal(loc = 0, scale = 1/numpy.sqrt(784) , size = (midnum, 784))
        w_two = normal(loc = 0, scale = 1/numpy.sqrt(midnum) , size = (10, midnum))
        b_one = normal(loc = 0, scale = 1/numpy.sqrt(784) , size = (midnum, 1))
        b_two = normal(loc = 0, scale = 1/numpy.sqrt(midnum) , size = (10, 1))
        
        gamma = numpy.ones((midnum, 1))
        beta = numpy.zeros((midnum, 1))
        running_mean = None
        running_var = None
    
    eponum = 1
    
    for i in range(int(eponum * (trainsize / 100))): #i <- N/B * 100 or so
        rannumlist = numpy.random.choice(trainsize, 100, replace=False)
        xlist = (dlist[rannumlist] / 255).T
        #xlist.shape=[784,100]
        ylist = anslist[rannumlist]
        #ylist.shape=[100,]
        #one-hot-vector-ize
        y_onehot = numpy.identity(10)[ylist]
        #y_onehot.shape=[100,10]
        
        normclass = Batchnorm(gamma, beta, batchmoment, running_mean, running_var)
        (midrslt, outrslt, loss_ave, new_running_mean, new_running_var) = \
                threeclass.three_nn(xlist, y_onehot, \
                                    w_one, b_one, w_two, b_two, \
                                    100, itbool, srclass, dropclass, normclass)
        print(loss_ave)
        
        (en_w_one, en_w_two, en_b_one, en_b_two, en_gamma, en_beta) = \
                backclass.back(xlist, midrslt, outrslt, y_onehot, w_one, w_two, srclass, dropclass, normclass)

        en_b_one = en_b_one.reshape(midnum,1)
        en_b_two = en_b_two.reshape(10,1)
    
        #update w and b
        lr = 0.01 #learn rate
        w_one = w_one - lr * en_w_one
        w_two = w_two - lr * en_w_two
        b_one = b_one - lr * en_b_one
        b_two = b_two - lr * en_b_two
        gamma = gamma - lr * en_gamma
        beta = beta - lr * en_beta
        running_mean = new_running_mean
        running_var = new_running_var
        
    wb = numpy.array([w_one, w_two, b_one, b_two, gamma, beta, running_mean, running_var])
    numpy.save(npyfile, wb)

else:
    innum = hyoinclass.hyojun_in()

    xlist = (dlist[innum] / 255).T
    ylist = anslist[innum]
    y_onehot = numpy.identity(10)[ylist]

    normclass = Batchnorm(gamma, beta, batchmoment, running_mean, running_var)
    outrslt = threeclass.three_nn(xlist, y_onehot, w_one, b_one, w_two, b_two, 1, itbool, srclass, dropclass, normclass)[1]

    rslt = numpy.argmax(outrslt)
    print(rslt, "is written in number", innum, "image, maybe.")
    print(ylist, "is written in number", innum, "image, definitely.")
