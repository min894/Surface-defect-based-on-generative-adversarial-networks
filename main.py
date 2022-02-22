#coding:utf-8
#daixiangyu

from data_set import datasets
import torch
from torch.utils.data import DataLoader
import model
import tools
from config import configs
import torch.nn.functional as F

global device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('use cuda')
else:
    device = torch.device("cpu")
    print('use cpu')


def train(D_net, G_net, loader):
    real_label = torch.randint(1, 2, (configs.batch_size,)).to(device)
    fake_label = torch.randint(0, 1, (configs.batch_size,)).to(device)
    d_loss_ = 0
    for epoch in range(configs.epochs):
        # 训练鉴别器
        for data in loader:
            real_x = data[0].to(device)
            fake_x = G_net(torch.randn((configs.batch_size, 100, 1, 1)).to(device)).detach()
            d_real_y = D_net.forward(real_x)
            d_fake_y = D_net.forward(fake_x)

            # 真实损失
            d_loss_real = F.cross_entropy(d_real_y, real_label)
            # 假损失
            d_loss_fake = F.cross_entropy(d_fake_y, fake_label)
            # 鉴别器损失
            d_loss = d_loss_real + d_loss_fake



            d_loss_ = d_loss.item()
            D_net.optim.zero_grad()
            d_loss.backward()
            D_net.optim.step()

        g_loss_ = 0
        for i in range(5):
            # 训练生成器
            noise = torch.randn((configs.batch_size, 100, 1, 1)).to(device)
            g_fake_x = G_net.forward(noise)
            g_fake_y = D_net.forward(g_fake_x)

            # 生成器损失
            g_loss = F.cross_entropy(g_fake_y, real_label)

            g_loss_ = g_loss.item()
            G_net.optim.zero_grad()
            g_loss.backward()
            G_net.optim.step()

        if epoch % 10 == 0:
            tools.log(epoch, g_loss_, d_loss_)
            test(G_net, epoch)

        if epoch % 10 == 0:
            G_net.save_model(path='models/g_model.pth')
            D_net.save_model(path='models/d_model.pth')


def test(G_net, step):
    noise = torch.randn((1, 100, 1, 1)).to(device)
    y = G_net.forward(noise).detach().cpu()
    tools.img_save(y, step)




if __name__ == '__main__':
    D_net = model.netD().to(device)
    G_net = model.netG().to(device)
    loader = DataLoader(
        dataset=datasets,
        batch_size=configs.batch_size,
        shuffle=True
    )
    train(D_net, G_net, loader)







