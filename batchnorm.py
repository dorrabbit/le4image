import numpy as np

class Batchnorm:
    def __init__(self, gamma, beta, momentum, running_mean, running_var):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var
        
        self.summid = None
        self.epsilon = 10e-7
        self.ave = None
        self.dispersion = None
        self.new_x = None
        
    def forward(self, summid, itbool):
        self.summid = summid
        self.batchsize = self.summid.shape[1]
        self.midnum = self.summid.shape[0]
        #summid.shape = [55,100]
        
        if not itbool:
            self.ave = (self.summid.sum(axis=1) / self.batchsize).reshape(self.midnum, 1)
            self.dispersion = ((np.power(self.summid - self.ave, 2)).sum(axis=1) / self.batchsize).reshape(self.midnum, 1)
            
            self.new_x = (self.summid - self.ave) / np.sqrt(self.dispersion + self.epsilon)
            new_y = self.gamma * self.new_x + self.beta

            #update mean and var
            this_mean = self.summid.mean(axis=1).reshape(self.midnum, 1)
            this_var = np.mean((self.summid - this_mean) ** 2, axis=1).reshape(self.midnum, 1)
            if self.running_mean is None: #first time
                self.running_mean = this_mean
                self.running_var = this_var
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * this_mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * this_var
            
            return (new_y, self.running_mean, self.running_var)
            #return new_y
        else:
            gamma_sqrt = self.gamma / np.sqrt(self.running_var + self.epsilon)
            new_y = gamma_sqrt * summid + (self.beta - gamma_sqrt * self.running_mean)

            return (new_y, None, None)

    def backward(self, dout):
        #dout.shape = [self.midnum,100]
        en_new_x  = (dout * self.gamma).reshape(self.midnum, self.batchsize)
       
        en_dispersion = ((en_new_x * (self.summid - self.ave)).sum(axis=1).reshape(self.midnum, 1) \
                         * (-1/2) \
                         * np.power(self.dispersion + self.epsilon, -3/2)).reshape(self.midnum, 1)
        
        en_ave = (en_new_x * (-1)/np.sqrt(self.dispersion + self.epsilon)).sum(axis=1).reshape(self.midnum, 1) \
                 + \
                 en_dispersion * ((-2)*(self.summid - self.ave)).sum(axis=1).reshape(self.midnum, 1) / self.batchsize
        
        en_summid = en_new_x * 1/np.sqrt(self.dispersion + self.epsilon) \
                    + en_dispersion * 2 * (self.summid - self.ave) / self.batchsize \
                    + en_ave / self.batchsize
        
        en_gamma = (dout * self.new_x).sum(axis=1).reshape(self.midnum, 1)
        en_beta = dout.sum(axis=1).reshape(self.midnum, 1)

        return (en_summid, en_gamma, en_beta)
