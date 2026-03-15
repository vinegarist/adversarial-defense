import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients


class OcclusionAttack(nn.Module):
    def __init__(self, net, top_k=9, occlu_color=0.0, kernel_size=3):
        """
        参数说明：
        net: 待攻击的模型
        top_k: 选择梯度积分值最大的前top_k个像素作为遮蔽中心
        occlu_color: 遮蔽颜色，0为黑色，1为白色，0.5为灰色
        kernel_size: 遮蔽窗口大小
        """
        super(OcclusionAttack, self).__init__()
        self.net = net
        self.top_k = top_k
        self.occlu_color = occlu_color
        self.kernel_size = kernel_size

    def forward(self, inputs):
        x, y = inputs
        device = x.device
        bs = x.shape[0]
        channels = x.shape[1]
        padding = self.kernel_size // 2

        with torch.enable_grad():
            x_ig = x.detach().requires_grad_()
            ig = IntegratedGradients(self.net)
            attr_ig = ig.attribute(x_ig, target=y, n_steps=50).detach().float()

        conv_sum = nn.Conv2d(channels, 1, kernel_size=self.kernel_size,
                             stride=1, padding=padding, bias=False).to(device)
        conv_sum.weight = nn.Parameter(
            torch.ones([1, channels, self.kernel_size, self.kernel_size],
                       dtype=torch.float32).to(device))
        conv_sum.weight.requires_grad_(False)

        out_sum_ig = conv_sum(attr_ig)

        out_sum_ig_sort = torch.sort(
            out_sum_ig.view(bs, -1), descending=True)[0]
        threshold = out_sum_ig_sort[:, self.top_k].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        pixel_mask = ((out_sum_ig - threshold) >= 0).float().to(device)

        conv_expand = nn.Conv2d(1, 1, kernel_size=self.kernel_size,
                                stride=1, padding=padding, bias=False).to(device)
        conv_expand.weight = nn.Parameter(
            torch.ones([1, 1, self.kernel_size, self.kernel_size],
                       dtype=torch.float32).to(device))
        conv_expand.weight.requires_grad_(False)

        region_mask = conv_expand(pixel_mask)
        region_mask_color = region_mask.repeat(1, channels, 1, 1)

        mask = (region_mask_color > 0)
        occlu = torch.ones_like(x) * self.occlu_color

        x_adv = torch.clamp((~mask) * x.detach() + mask * occlu, min=0, max=1)

        return x_adv
