import numpy as np
from relu import Relu

class convolutional:
    def __init__(self, w, b):
        self.fsize_x = 3
        self.fsize_y = 3
        self.fsize_z = 1
        self.k = 20
        self.w = w
        #w.shape = (20, 3*3*1)
        self.b = b
        self.x = None
        #x.shape = (3*3*1, 28*28*100)
        self.reluclass = Relu()
        
    def forward(self, image):
        #image.shape(100, 1, 28, 28)

        (batchnum, ch, height, width) = image.shape
        image = np.pad(image, [(0,0),(0,0),(1,1),(1,1)])
        #image.shape = (100, 1, 30, 30)

        self.x = np.zeros((batchnum, ch, 3, 3, height, width))

        for y in range(3):
            y_max = y + height
            for x in range(3):
                x_max = x + width
                self.x[:, :, y, x, :, :] = image[:, :, y:y_max, x:x_max]

        self.x = self.x.transpose(0, 4, 5, 1, 2, 3).reshape(batchnum*height*width, -1)

        
        y = np.dot(self.w, self.x) + self.b
        #y.shape = (20, )
        
        convrslt = reluclass.forward(convrslt)
