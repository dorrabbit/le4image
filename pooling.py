import numpy as np

class Pooling:
    def __init__(self):
        self.x = None
        self.y = None
        
    def forward(self, image):
        #image.shape = (20*28*28, 100)
        self.x = image.T.reshape(-1, 20, 28, 28)
        batchsize = self.x.shape[0]

        x_ = np.zeros((batchsize, 20, 2, 2, 14, 14))

        for y in range(2):
            y_max = y + 14
            for x in range(2):
                x_max = x + width
                x_[:, :, y, x, :, :] = self.x[:, :, y:y_max, x:x_max]

        x_ = x_.transpose(0, 4, 5, 1, 2, 3).reshape(-1, 2*2)
        out = np.max(x_, axis=1)

        out = out.reshape(batchsize, 14, 14, 20).transpose(3, 1, 2, 0).reshape(-1, batchsize)
        return out

    def backward(self, en_y):
        
