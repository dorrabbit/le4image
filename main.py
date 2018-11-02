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

[w_one, w_two, b_one, b_two, gamma_mid, beta_mid, gamma_out, beta_out,\
 running_mean_mid, running_var_mid, running_mean_out, running_var_out] = numpy.load(npyfile)
#w_one = numpy.load(npyfile)[0]
#w_two = numpy.load(npyfile)[1]
#b_one = numpy.load(npyfile)[2]
#b_two = numpy.load(npyfile)[3]

if not itbool:
    load_yn = input("\r" + "use saved weight? y/n > ")
    if load_yn == "n":
        #initialize
        w_one = normal(loc = 0, scale = 1/numpy.sqrt(784) , size = (midnum, 784))
        w_two = normal(loc = 0, scale = 1/numpy.sqrt(midnum) , size = (10, midnum))
        b_one = normal(loc = 0, scale = 1/numpy.sqrt(784) , size = (midnum, 1))
        b_two = normal(loc = 0, scale = 1/numpy.sqrt(midnum) , size = (10, 1))
        
        gamma_mid = numpy.ones((midnum, 1))
        beta_mid = numpy.zeros((midnum, 1))
        gamma_out = numpy.ones((10, 1))
        beta_out = numpy.zeros((10, 1))
        running_mean_mid = None
        running_var_mid = None
        running_mean_out = None
        running_var_out = None
    
    eponum = 1

    #initialize
    pre_w_one_update = 0
    pre_w_two_update = 0

    #veriable
    lr = 0.01 #learn rate
    alpha = 0.9
    
    for i in range(int(eponum * (trainsize / 100))): #i <- N/B * 100 or so
        rannumlist = numpy.random.choice(trainsize, 100, replace=False)
        xlist = (dlist[rannumlist] / 255).T
        #xlist.shape=[784,100]
        ylist = anslist[rannumlist]
        #ylist.shape=[100,]
        #one-hot-vector-ize
        y_onehot = numpy.identity(10)[ylist]
        #y_onehot.shape=[100,10]
        
        normclass_mid = Batchnorm(gamma_mid, beta_mid, batchmoment, running_mean_mid, running_var_mid)
        normclass_out = Batchnorm(gamma_out, beta_out, batchmoment, running_mean_out, running_var_out)
        (midrslt, outrslt, loss_ave, \
         new_running_mean_mid, new_running_var_mid, new_running_mean_out, new_running_var_out) = \
                threeclass.three_nn(xlist, y_onehot, \
                                    w_one, b_one, w_two, b_two, \
                                    100, itbool, srclass, dropclass, normclass_mid, normclass_out)
        print(loss_ave)
        
        (en_w_one, en_w_two, en_b_one, en_b_two, en_gamma_mid, en_beta_mid, en_gamma_out, en_beta_out) = \
                backclass.back(xlist, midrslt, outrslt, y_onehot, w_one, w_two, \
                               srclass, dropclass, normclass_mid, normclass_out)

        en_b_one = en_b_one.reshape(midnum,1)
        en_b_two = en_b_two.reshape(10,1)
    
        #update w and b
        w_one_update = lr * en_w_one
        w_one = w_one + alpha * pre_w_one_update - w_one_update
        w_two_update = lr * en_w_two
        w_two = w_two + alpha * pre_w_two_update - w_two_update
        b_one = b_one - lr * en_b_one
        b_two = b_two - lr * en_b_two
        gamma_mid = gamma_mid - lr * en_gamma_mid
        beta_mid = beta_mid - lr * en_beta_mid
        gamma_out = gamma_out - lr * en_gamma_out
        beta_out = beta_out - lr * en_beta_out
        running_mean_mid = new_running_mean_mid
        running_var_mid = new_running_var_mid
        running_mean_out = new_running_mean_out
        running_var_out = new_running_var_out

        #update pre_w_update
        pre_w_one_update = w_one_update
        pre_w_two_update = w_two_update
        
    wb = numpy.array([w_one, w_two, b_one, b_two, \
                      gamma_mid, beta_mid, gamma_out, beta_out, \
                      running_mean_mid, running_var_mid, running_mean_out, running_var_out])
    numpy.save(npyfile, wb)

else:
    innum = hyoinclass.hyojun_in()

    xlist = (dlist[innum] / 255).T
    ylist = anslist[innum]
    y_onehot = numpy.identity(10)[ylist]

    normclass_mid = Batchnorm(gamma_mid, beta_mid, batchmoment, running_mean_mid, running_var_mid)
    normclass_out = Batchnorm(gamma_out, beta_out, batchmoment, running_mean_out, running_var_out)
    outrslt = threeclass.three_nn(xlist, y_onehot, w_one, b_one, w_two, b_two, \
                                  1, itbool, srclass, dropclass, normclass_mid, normclass_out)[1]

    rslt = numpy.argmax(outrslt)
    print(rslt, "is written in number", innum, "image, maybe.")
    print(ylist, "is written in number", innum, "image, definitely.")
