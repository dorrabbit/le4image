class Inputer:
    def inputer(self):
        import numpy as np
        from mnist import MNIST
        mndata = MNIST("/export/home/016/a0169573/le4/image_process/le4nn/")
        X, Y = mndata.load_training()
        X = np.array(X)
        #X = X.reshape((X.shape[0],28,28))
        Y = np.array(Y)
        return (X,Y)
