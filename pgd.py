import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ========== 辅助函数（参考 Inuput-Gradient-Distillation）==========
def norms_l0(Z):
    """计算L0范数（非零元素个数）"""
    return ((Z.view(Z.shape[0], -1) != 0).sum(dim=1)[:, None, None, None]).float()


def norms_l1(Z):
    """计算L1范数"""
    return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None]


def norms_l2(Z):
    """计算L2范数"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


def norms_linf(Z):
    """计算L∞范数"""
    return Z.view(Z.shape[0], -1).abs().max(dim=1)[0]


def cwloss(output, target, confidence=50, num_classes=10):
    """Carlini-Wagner损失函数

    参数说明：
    output: 模型输出logits
    target: 真实标签
    confidence: 置信度边际参数
    num_classes: 类别数
    """
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,)).to(output.device)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)

    real = (target_onehot * output).sum(1)
    other = ((1. - target_onehot) * output - target_onehot * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)
    loss = torch.sum(loss)
    return loss


# ========== L∞ PGD 攻击（原有）==========
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


# ========== L2 PGD 攻击（参考 _pgd_whitebox_l2）==========
class L2PGD(nn.Module):
    """L2范数PGD攻击

    与L∞ PGD的区别：
    - 扰动约束使用L2范数而非L∞范数
    - 梯度需要归一化后再更新
    """

    def __init__(self,
                 net,
                 eps=1.0,
                 step_size=0.1,
                 step=10,
                 random_start=True,
                 criterion=F.cross_entropy):
        """
        参数说明：
        net: 待攻击的模型
        eps: L2范数扰动上限
        step_size: 每步扰动大小
        step: 迭代次数
        random_start: 是否随机初始化
        criterion: 损失函数
        """
        super(L2PGD, self).__init__()
        self.net = net
        self.eps = eps
        self.step_size = step_size
        self.step = step
        self.random_start = random_start
        self.criterion = criterion

    def forward(self, inputs):
        x, y = inputs
        bs = x.shape[0]

        if self.random_start:
            # 使用小随机噪声初始化
            x_adv = x + 0.001 * torch.randn_like(x)
            x_adv = torch.clamp(x_adv, 0., 1.)
        else:
            x_adv = x.clone()

        with torch.enable_grad():
            for _ in range(self.step):
                x_adv = x_adv.requires_grad_()
                logits = self.net(x_adv)
                loss = self.criterion(logits, y) * bs

                grad = torch.autograd.grad(loss, x_adv)[0]
                # 梯度归一化（L2方向）
                grad_norms = grad.view(bs, -1).norm(p=2, dim=1).view(bs, 1, 1, 1)
                # 避免除零
                grad_norms = grad_norms.clamp(min=1e-12)
                grad_normalized = grad / grad_norms

                # 更新扰动
                x_adv = x_adv + self.step_size * grad_normalized

                # L2投影：将扰动限制在eps球内
                delta = x_adv - x
                delta_norms = delta.view(bs, -1).norm(p=2, dim=1).view(bs, 1, 1, 1)
                delta = delta * (delta_norms.clamp(max=self.eps) / delta_norms.clamp(min=1e-12))
                x_adv = torch.clamp(x + delta, 0., 1.).clone().detach()

        return x_adv


# ========== FGSM 攻击（参考 _fgsm_whitebox）==========
class FGSM(nn.Module):
    """Fast Gradient Sign Method (FGSM) 攻击

    单步L∞ PGD攻击的特例
    """

    def __init__(self, net, eps=8/255, criterion=F.cross_entropy):
        """
        参数说明：
        net: 待攻击的模型
        eps: 扰动大小
        criterion: 损失函数
        """
        super(FGSM, self).__init__()
        self.net = net
        self.eps = eps
        self.criterion = criterion

    def forward(self, inputs):
        x, y = inputs
        bs = x.shape[0]

        with torch.enable_grad():
            x_adv = x.clone().requires_grad_()
            logits = self.net(x_adv)
            loss = self.criterion(logits, y) * bs

            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv + self.eps * grad.sign()
            x_adv = torch.clamp(x_adv, 0., 1.)

        return x_adv


# ========== CW 攻击（参考 _cw_whitebox）==========
class CWAttack(nn.Module):
    """Carlini-Wagner L∞攻击

    使用CW损失函数的迭代攻击
    """

    def __init__(self, net, eps=8/255, step_size=2/255, step=10,
                 confidence=50, num_classes=10, random_start=True):
        """
        参数说明：
        net: 待攻击的模型
        eps: L∞扰动上限
        step_size: 每步扰动大小
        step: 迭代次数
        confidence: CW损失的置信度参数
        num_classes: 类别数
        random_start: 是否随机初始化
        """
        super(CWAttack, self).__init__()
        self.net = net
        self.eps = eps
        self.step_size = step_size
        self.step = step
        self.confidence = confidence
        self.num_classes = num_classes
        self.random_start = random_start

    def forward(self, inputs):
        x, y = inputs
        bs = x.shape[0]

        up = torch.clamp(x + self.eps, 0., 1.)
        down = torch.clamp(x - self.eps, 0., 1.)

        if self.random_start:
            t = 2 * torch.rand(x.shape).to(x.device).detach() - 1
            x_adv = x + self.eps * t
        else:
            x_adv = x.clone()

        x_adv = torch.max(torch.min(x_adv, up), down)

        with torch.enable_grad():
            for _ in range(self.step):
                x_adv = x_adv.requires_grad_()
                logits = self.net(x_adv)
                loss = cwloss(logits, y, confidence=self.confidence, num_classes=self.num_classes)

                grad = torch.autograd.grad(loss, x_adv)[0]
                x_adv = x_adv + self.step_size * grad.sign()
                x_adv = torch.max(torch.min(x_adv, up), down).clone().detach()

        return x_adv


# ========== Multi-Scale Diversity 攻击（参考 msd_v0）==========
class MSDAttack(nn.Module):
    """Multi-Scale Diversity攻击

    同时使用L2和L∞两种扰动方式，选择攻击效果更好的方向
    """

    def __init__(self, net,
                 epsilon_l_inf=8/255,
                 epsilon_l_2=1.0,
                 alpha_l_inf=2/255,
                 alpha_l_2=0.1,
                 num_iter=10):
        """
        参数说明：
        net: 待攻击的模型
        epsilon_l_inf: L∞扰动上限
        epsilon_l_2: L2扰动上限
        alpha_l_inf: L∞每步扰动大小
        alpha_l_2: L2每步扰动大小
        num_iter: 迭代次数
        """
        super(MSDAttack, self).__init__()
        self.net = net
        self.epsilon_l_inf = epsilon_l_inf
        self.epsilon_l_2 = epsilon_l_2
        self.alpha_l_inf = alpha_l_inf
        self.alpha_l_2 = alpha_l_2
        self.num_iter = num_iter

    def forward(self, inputs):
        x, y = inputs
        bs = x.shape[0]

        delta = torch.zeros_like(x, requires_grad=False)
        max_delta = torch.zeros_like(x)
        max_loss = torch.zeros(y.shape[0]).to(y.device)

        for t in range(self.num_iter):
            delta.requires_grad_(True)

            with torch.enable_grad():
                logits = self.net(x + delta)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                grad = delta.grad.data

            delta.requires_grad_(False)

            with torch.no_grad():
                # L2方向更新
                grad_norm = norms_l2(grad).clamp(min=1e-12)
                delta_l2 = delta.data + self.alpha_l_2 * grad / grad_norm
                # L2投影
                delta_l2_norm = norms_l2(delta_l2).clamp(min=1e-12)
                delta_l2 = delta_l2 * torch.clamp(self.epsilon_l_2 / delta_l2_norm, max=1.0)
                # 限制在有效像素范围
                delta_l2 = torch.clamp(delta_l2, -x, 1 - x)

                # L∞方向更新
                delta_linf = delta.data + self.alpha_l_inf * grad.sign()
                delta_linf = delta_linf.clamp(-self.epsilon_l_inf, self.epsilon_l_inf)
                delta_linf = torch.clamp(delta_linf, -x, 1 - x)

                # 比较两种扰动的攻击效果
                for delta_temp in [delta_l2, delta_linf]:
                    loss_temp = F.cross_entropy(self.net(x + delta_temp), y, reduction='none')
                    max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
                    max_loss = torch.max(max_loss, loss_temp)

                # 更新delta
                delta = max_delta.clone()

        return torch.clamp(x + max_delta, 0., 1.)