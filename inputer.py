class Inputer:
    def inputer(self, itbool):
        import numpy as np
        from mnist import MNIST
        mndata = MNIST("/export/home/016/a0169573/le4/image_process/le4nn/")

        if itbool:
            X, Y = mndata.load_testing()
        else:
            X, Y = mndata.load_training()
            
        X = np.array(X)
        X = X.reshape(-1, 1, 28, 28)
        Y = np.array(Y)
        return (X,Y)
