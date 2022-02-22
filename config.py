#coding:utf-8


#超参数
class config(object):
    def __init__(self):
        self.data_path = 'MEMS100/'

        # train
        self.batch_size = 10
        self.lr = 0.00003
        self.epochs = 10000

        # net
        self.noise_dim = 100 #(100,)
        self.ngf = 64

        # savepath
        self.path_g = 'model/gnet.pth'
        self.path_d = 'model/dnet.pth'

        self.imgsize = 1024

configs = config()