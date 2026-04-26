"""
Fixed Saliency-based Occlusion Attacks

Key Fix: Use Loss-based gradient instead of Score-based gradient
- Original: grad(output[label], x) -> finds pixels that make model more confident
- Fixed: grad(Loss, x) -> finds pixels that make model WRONG (correct for occlusion attack)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_saliency_loss_based(model, x, y):
    """计算Loss-based梯度归因图（修复版）

    与原版的区别：
    - 原版: torch.autograd.grad(output[label], x) -> Score-based
    - 修复版: torch.autograd.grad(Loss, x) -> Loss-based

    参数:
        model: 神经网络模型
        x: 输入图像 [B, C, H, W]，需要requires_grad
        y: 标签 [B,]

    返回:
        saliency_map: 梯度归因图 [B, C, H, W]
    """
    model.eval()
    output = model(x)
    # 关键修改：使用CrossEntropyLoss而非output[label]
    loss = F.cross_entropy(output, y)
    saliency_map = torch.autograd.grad(loss, x, torch.ones_like(loss))[0]
    return saliency_map.abs()


def compute_saliency_score_based(model, x, y):
    """原版Score-based梯度归因图（保留用于对比）"""
    model.eval()
    output = model(x)
    scores = output.gather(1, y.view(-1, 1))  # 取正确类别的分数
    saliency_map = torch.autograd.grad(scores, x, torch.ones_like(scores))[0]
    return saliency_map.abs()


class SaliencyOcclusionAttack_Fixed(nn.Module):
    """修复版：基于Saliency（Loss-based梯度）的固定遮蔽攻击"""

    def __init__(self, net, top_k=9, occlu_color=0.0, kernel_size=3, loss_based=True):
        """
        参数说明：
        net: 待攻击的模型
        top_k: 选择梯度值最大的前top_k个像素作为遮蔽中心
        occlu_color: 遮蔽颜色，0为黑色，1为白色，0.5为灰色
        kernel_size: 遮蔽窗口大小
        loss_based: True=修复版(Loss梯度), False=原版(Score梯度)
        """
        super(SaliencyOcclusionAttack_Fixed, self).__init__()
        self.net = net
        self.top_k = top_k
        self.occlu_color = occlu_color
        self.kernel_size = kernel_size
        self.loss_based = loss_based

    def forward(self, inputs):
        x, y = inputs
        device = x.device
        bs = x.shape[0]
        channels = x.shape[1]
        H, W = x.shape[2], x.shape[3]
        padding = self.kernel_size // 2

        # 使用修复版或原版梯度计算
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            if self.loss_based:
                attr_map = compute_saliency_loss_based(self.net, x_grad, y)
            else:
                attr_map = compute_saliency_score_based(self.net, x_grad, y)

        # 对多通道求和
        if channels > 1:
            attr_sum = attr_map.sum(dim=1, keepdim=True)
        else:
            attr_sum = attr_map

        # 使用卷积计算每个位置的kernel_size邻域的梯度总和
        conv_sum = nn.Conv2d(1, 1, kernel_size=self.kernel_size,
                             stride=1, padding=padding, bias=False).to(device)
        conv_sum.weight = nn.Parameter(
            torch.ones([1, 1, self.kernel_size, self.kernel_size],
                       dtype=torch.float32).to(device))
        conv_sum.weight.requires_grad_(False)

        out_sum = conv_sum(attr_sum)

        # 找到top_k个最重要的区域
        out_sum_flat = out_sum.view(bs, -1)
        _, top_indices = torch.topk(out_sum_flat, self.top_k, dim=1)

        # 创建遮蔽掩码
        mask = torch.zeros(bs, 1, H, W, device=device)
        for i in range(bs):
            for idx in top_indices[i]:
                row = idx // W
                col = idx % W
                r_start = max(0, row - padding)
                r_end = min(H, row + padding + 1)
                c_start = max(0, col - padding)
                c_end = min(W, col + padding + 1)
                mask[i, 0, r_start:r_end, c_start:c_end] = 1

        # 应用遮蔽
        mask = mask.repeat(1, channels, 1, 1)
        occlu = torch.ones_like(x) * self.occlu_color
        x_adv = torch.clamp((1 - mask) * x.detach() + mask * occlu, 0, 1)

        return x_adv


class AdaptiveSaliencyOcclusionAttack_Fixed(nn.Module):
    """修复版：基于Saliency（Loss-based梯度）的自适应遮蔽攻击"""

    def __init__(self, net, N=5, R=3, c=0.0, loss_based=True):
        """
        参数说明：
        net: 待攻击的模型
        N: 最大遮蔽区域数量
        R: 最大遮蔽半径
        c: 遮蔽颜色值
        loss_based: True=修复版(Loss梯度), False=原版(Score梯度)
        """
        super(AdaptiveSaliencyOcclusionAttack_Fixed, self).__init__()
        self.net = net
        self.N = N
        self.R = R
        self.c = c
        self.loss_based = loss_based

    def forward(self, inputs):
        x, y = inputs
        device = x.device
        bs = x.shape[0]
        channels = x.shape[1]
        H, W = x.shape[2], x.shape[3]

        # 使用修复版或原版梯度计算
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            if self.loss_based:
                attr_map = compute_saliency_loss_based(self.net, x_grad, y)
            else:
                attr_map = compute_saliency_score_based(self.net, x_grad, y)

        # 对多通道求和后展平排序
        if channels > 1:
            attr_flat = attr_map.sum(dim=1).view(bs, -1)
        else:
            attr_flat = attr_map.view(bs, -1)

        _, max_region_index = torch.sort(attr_flat, dim=1, descending=True)

        # 预计算所有 (n, r) 组合的累积遮蔽掩码
        nr2mask = {}
        with torch.no_grad():
            for n in range(1, self.N + 1):
                for r in range(1, self.R + 1):
                    nr2mask[f'{n}_{r}'] = torch.zeros(bs, channels, H, W, device=device)

            for sample_idx in range(bs):
                for r in range(1, self.R + 1):
                    for i in range(self.N):
                        region_index = max_region_index[sample_idx, i]
                        selected_i = region_index.item() // W
                        selected_j = region_index.item() % W

                        left_x = max(selected_i - r, 0)
                        left_y = max(selected_j - r, 0)
                        right_x = min(selected_i + r + 1, H)
                        right_y = min(selected_j + r + 1, W)

                        nr2mask[f'{i+1}_{r}'][sample_idx, :, left_x:right_x, left_y:right_y] = 1

                        if i > 0:
                            nr2mask[f'{i+1}_{r}'][sample_idx] = torch.clamp(
                                nr2mask[f'{i+1}_{r}'][sample_idx] + nr2mask[f'{i}_{r}'][sample_idx],
                                0, 1
                            )

        # 渐进式遮蔽 + 逐样本早停
        with torch.no_grad():
            pred_init = self.net(x).argmax(dim=1)
        sample2perturb = (pred_init.cpu() == y.cpu()).numpy()

        occ_x = x.clone()

        with torch.no_grad():
            for n in range(1, self.N + 1):
                if sample2perturb.sum() == 0:
                    break
                for r in range(1, self.R + 1):
                    if sample2perturb.sum() == 0:
                        break

                    mask = nr2mask[f'{n}_{r}']

                    for idx in range(bs):
                        if not sample2perturb[idx]:
                            continue
                        occ_x[idx] = (1 - mask[idx]) * x[idx] + mask[idx] * self.c

                    occ_x = torch.clamp(occ_x, 0, 1)

                    pred = self.net(occ_x).argmax(dim=1)
                    sample2perturb = (pred.cpu() == y.cpu()).numpy()

        return occ_x
