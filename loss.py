import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, reduction='mean'):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, y, target):
        # 获取总类别数
        c = y.shape[1]
        
        # 将target转化为label smoothing的soft label
        # 1) 将target转化为one-hot形式的概率向量，此处torch.scatter函数在全0向量的指定位置写入1.，
        # 参考https://pytorch.org/docs/1.7.1/tensors.html?highlight=torch%20scatter#torch.Tensor.scatter
        soft_one_hot = torch.scatter(torch.zeros_like(y), dim=1, index=target.unsqueeze(1), value=1.)
        # 2) 定义均匀分布概率向量
        soft_uniform = 1/c * torch.ones_like(y)
        # 3) 计算label smoothing soft label
        soft_label = self.label_smoothing * soft_uniform + (1 - self.label_smoothing) * soft_one_hot

        # 计算交叉熵损失
        loss = (-F.log_softmax(y, dim=1) * soft_label).sum(dim=1)
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError


# 定义CW损失
def CWLoss(logits, targets, kappa=0, targeted=False, reduction='mean'):
    y_onehot = torch.nn.functional.one_hot(targets, 10).to(torch.float)

    real = torch.sum(y_onehot * logits, dim=1)
    other, _ = torch.max((1 - y_onehot) * logits - y_onehot * 1e4, dim=1)

    # 有些代码中使用relu()代替，效果相同
    if targeted:
        loss = torch.clamp(other - real + kappa, 0)
    # 若想实现非目标攻击，只需传入真实标签，将(other - real)改为(real - other)即可
    else:
        loss = torch.clamp(real - other + kappa, 0)
    
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    elif reduction == 'none':
        pass
    else:
        raise NotImplementedError

    return -loss