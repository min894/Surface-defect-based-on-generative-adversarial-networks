#coding:utf-8


import torch.nn as nn
from config import configs
import torch

class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()
        self.layer = nn.Sequential(


            # (100,1,1) to (800, 4,4)
            nn.ConvTranspose2d(in_channels=configs.noise_dim, out_channels=configs.ngf * 16,
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.InstanceNorm2d(configs.ngf * 16),
            nn.ReLU(inplace=True),

            # (512, 4, 4) to (256, 16, 16)
            nn.ConvTranspose2d(in_channels=configs.ngf * 16, out_channels=configs.ngf * 8,
                               kernel_size=4,
                               stride=4,
                               padding=0,
                               bias=False),
            nn.InstanceNorm2d(configs.ngf * 8),
            nn.ReLU(inplace=True),

            # (512, 16, 16) to (128, 64, 64)
            nn.ConvTranspose2d(in_channels=configs.ngf * 8, out_channels=configs.ngf * 4,
                               kernel_size=4,
                               stride=4,
                               padding=0,
                               bias=False),
            nn.InstanceNorm2d(configs.ngf * 4),
            nn.ReLU(inplace=True),

            # (128, 64, 64) to (128, 256, 256)
            nn.ConvTranspose2d(in_channels=configs.ngf * 4, out_channels=configs.ngf * 2,
                               kernel_size=4,
                               stride=4,
                               padding=0,
                               bias=False),
            nn.InstanceNorm2d(configs.ngf * 2),
            nn.ReLU(inplace=True),

            # (128, 256, 256) to (128, 1024, 1024)
            nn.ConvTranspose2d(in_channels=configs.ngf * 2, out_channels=3,
                               kernel_size=4,
                               stride=4,
                               padding=0,
                               bias=False),
            nn.Tanh(),

            # # (400, 12, 12)
            # nn.ConvTranspose2d(configs.ngf * 8, configs.ngf * 4, 6, 2, 0, bias=False),
            # nn.InstanceNorm2d(configs.ngf * 4),
            # nn.ReLU(inplace=True),
            #
            # # (200, 25, 25)
            # nn.ConvTranspose2d(configs.ngf * 4, configs.ngf * 2, 5, 2, 1, bias=False),
            # nn.InstanceNorm2d(configs.ngf * 2),
            # nn.ReLU(inplace=True),
            #
            # # (100, 50, 50)
            # nn.ConvTranspose2d(configs.ngf * 2, configs.ngf, 4, 2, 1, bias=False),
            # nn.InstanceNorm2d(configs.ngf),
            # nn.ReLU(inplace=True),

            # # (100, 150, 150)
            # nn.ConvTranspose2d(configs.ngf, 3, 5, 3, 1, bias=False),
            # nn.Tanh()
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=configs.lr)
        self.load_model('models/g_model.pth')

    def forward(self, x):
        x = self.layer(x)
        x = x * 128 + 128
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        try:
            self.load_state_dict(torch.load(path))
            print('gnet load model')
        except:
            print('gnet no model')


class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.layer1 = nn.Sequential(
            # input = (3, 150, 150)
            nn.Conv2d(3, configs.ngf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(configs.ngf, configs.ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(configs.ngf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(configs.ngf * 2, configs.ngf * 4, 5, 2, 1, bias=False),
            nn.InstanceNorm2d(configs.ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(configs.ngf * 4, configs.ngf * 8, 6, 2, 0, bias=False),
            nn.InstanceNorm2d(configs.ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(configs.ngf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512 * 40 * 40, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.optim = torch.optim.Adam(self.parameters(), lr=configs.lr)
        self.load_model('models/d_model.pth')

    def forward(self, x):
        x = self.layer1(x)
        x = x.reshape((-1, 512 * 40 * 40))
        x = self.layer2(x)
        return x

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        try:
            self.load_state_dict(torch.load(path))
            print('dnet load model')
        except:
            print('dnet no model')

