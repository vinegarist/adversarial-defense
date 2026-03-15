import torch
import torch.nn as nn
import torch.nn.functional as F


class LinfPGD(nn.Module):
    def __init__(self,
                 net,
                 eps=8 / 255,
                 step_size=2 / 255,
                 step=10,
                 random_start=True,
                 criterion=F.cross_entropy):
        """
        参数说明：
        net: 待攻击的模型
        eps, step_size, step: 攻击的迭代参数
        random_start: 控制攻击迭代开始前是否添加随机扰动
        criterion: 攻击所使用的损失函数

            LinfPGD类的默认构造参数实现的是10步的Linf PGD攻击，但正如在设计之初的
        所提到的那样，LinfPGD类可以同时实现FGSM, PGD, CW三种攻击，比如：
         - 设置参数step=1, random_start=False，并使用相同的step_size和eps，以实
           现FGSM攻击
         - 设置参数criterion为CW损失，以实现CW攻击
        """
        super(LinfPGD, self).__init__()
        self.net = net
        self.eps = eps
        self.step_size = step_size
        self.step = step
        self.random_start = random_start
        self.criterion = criterion

    def forward(self, inputs):
        x, y = inputs
        bs = x.shape[0]
        
        up = torch.clamp(x + self.eps, 0., 1.)
        down = torch.clamp(x - self.eps, 0., 1.)

        if self.random_start:
            t = 2 * torch.rand(x.shape).to(x.device).detach() - 1
            x_adv = x + self.eps * t
        else:
            x_adv = x

        x_adv = torch.max(torch.min(x_adv, up), down)
        with torch.enable_grad():
            for _ in range(self.step):
                x_adv = x_adv.requires_grad_()
                logits = self.net(x_adv)
                # 通常，我们默认以mean作为loss的reduction，这种情况下每个样本的梯度隐式地
                # 除以了batch size，因此此处乘以bs作为修正
                loss = self.criterion(logits, y) * bs
                grad = torch.autograd.grad(loss, x_adv)[0]
                x_adv = x_adv + self.step_size * torch.sign(grad)
                x_adv = torch.max(torch.min(x_adv, up), down).clone().detach()

        return x_adv