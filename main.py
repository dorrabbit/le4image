import numpy
from inputer import Inputer
inclass = Inputer()
from three_nn import Three_nn
threeclass = Three_nn()
from numpy.random import *
from back import Back
backclass = Back()
from hyojun_in import Hyojun_in
from sigmoid import Sigmoid
from relu import Relu
from batchnorm import Batchnorm
batchmoment = 0.9
from dropout import Dropout
dropclass = Dropout()
import sys
from adam import Adam

####
while True:
    iden_train = input("identification or training? i/t > ")
    if iden_train == "i":
        itbool = True
        break
    elif iden_train == "t":
        itbool = False
        break
    else:
        print("illegal input from keyboard.")
####

####
while True:
    sig_relu = input("sigmoid or relu? s/r > ")
    if sig_relu == "s":
        npyfile = 'wb_learn_s.npy'
        srclass = Sigmoid()
        break
    elif sig_relu == "r":
        npyfile = 'wb_learn_r.npy'
        srclass = Relu()
        break
    else:
        print("illegal input from keyboard.")
####
sys.stdout.write("Now loading...")
sys.stdout.flush()

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
    
    eponum = 100

    #initialize
    pre_w_one_update = 0
    pre_w_two_update = 0

    #update_variable
    #lr = 0.01 #learn rate
    w_one_adamclass = Adam(w_one.shape)
    b_one_adamclass = Adam(b_one.shape)
    w_two_adamclass = Adam(w_two.shape)
    b_two_adamclass = Adam(b_two.shape)
    gamma_mid_adamclass = Adam(gamma_mid.shape)
    beta_mid_adamclass = Adam(beta_mid.shape)
    gamma_out_adamclass = Adam(gamma_out.shape)
    beta_out_adamclass = Adam(beta_out.shape)
    
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
        w_one = w_one - w_one_adamclass.update_amount(en_w_one)
        b_one = b_one - b_one_adamclass.update_amount(en_b_one)
        w_two = w_two - w_two_adamclass.update_amount(en_w_two)
        b_two = b_two - b_two_adamclass.update_amount(en_b_two)
        gamma_mid = gamma_mid - gamma_mid_adamclass.update_amount(en_gamma_mid)
        beta_mid = beta_mid - beta_mid_adamclass.update_amount(en_beta_mid)
        gamma_out = gamma_out - gamma_out_adamclass.update_amount(en_gamma_out)
        beta_out = beta_out - beta_out_adamclass.update_amount(en_beta_out)
        
        running_mean_mid = new_running_mean_mid
        running_var_mid = new_running_var_mid
        running_mean_out = new_running_mean_out
        running_var_out = new_running_var_out
        
    wb = numpy.array([w_one, w_two, b_one, b_two, \
                      gamma_mid, beta_mid, gamma_out, beta_out, \
                      running_mean_mid, running_var_mid, running_mean_out, running_var_out])
    numpy.save(npyfile, wb)

else:
    while True:
        one_or_measure = input("\r" + "one-image-identification or accuracy-measurement? 1/% > ")
        if one_or_measure == "1":
            innum = Hyojun_in.hyojun_in()
            batchsize = 1
            break
        elif one_or_measure == "%":
            innum = numpy.random.choice(trainsize, 10000, replace=True)
            batchsize = 10000
            break
        else:
            print("illegal input from keyboard.")
        
    xlist = (dlist[innum] / 255).T
    ylist = anslist[innum]
    y_onehot = numpy.identity(10)[ylist]

    normclass_mid = Batchnorm(gamma_mid, beta_mid, batchmoment, running_mean_mid, running_var_mid)
    normclass_out = Batchnorm(gamma_out, beta_out, batchmoment, running_mean_out, running_var_out)
    outrslt = threeclass.three_nn(xlist, y_onehot, w_one, b_one, w_two, b_two, \
                                  batchsize, itbool, srclass, dropclass, normclass_mid, normclass_out)[1]
    rslt = numpy.argmax(outrslt, axis=0)
    
    if one_or_measure == "1":
        print(rslt[0], "is written in number", innum, "image, maybe.")
        print(ylist, "is written in number", innum, "image, definitely.")
        print("my identification is", rslt[0] == ylist)
    elif one_or_measure == "%":
        checkanslist = rslt == ylist
        print("accuracy is", numpy.sum(checkanslist) / 100, "%.")
