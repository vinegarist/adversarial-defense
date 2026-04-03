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


class AdaptiveOcclusionAttack(nn.Module):
    """渐进式自适应遮蔽攻击（参考 inductiveOcclusionAttack）。

    与 OcclusionAttack 的区别：
    - 逐步增加遮蔽区域数量(n=1..N)和遮蔽半径(r=1..R)
    - 对每个样本独立判断：一旦攻击成功（预测改变）即停止遮蔽
    - 因此使用最小遮蔽量，保留数字的语义可辨识性
    """

    def __init__(self, net, N=5, R=3, c=0.0):
        """
        参数说明：
        net: 待攻击的模型
        N: 最大遮蔽区域数量（逐步从1增加到N）
        R: 最大遮蔽半径（逐步从1增加到R）
        c: 遮蔽颜色值，0为黑色，1为白色
        """
        super(AdaptiveOcclusionAttack, self).__init__()
        self.net = net
        self.N = N
        self.R = R
        self.c = c

    def forward(self, inputs):
        x, y = inputs
        device = x.device
        bs = x.shape[0]
        channels = x.shape[1]
        H, W = x.shape[2], x.shape[3]

        # 1. 计算归因图（IntegratedGradients）
        with torch.enable_grad():
            x_ig = x.detach().requires_grad_()
            ig = IntegratedGradients(self.net)
            attr_ig = ig.attribute(x_ig, target=y, n_steps=50).detach().float()

        # 2. 将归因图展平并按重要性排序，找到每个样本的top-N个像素中心
        regional_attr = attr_ig.view(bs, -1)  # [bs, C*H*W]
        _, max_region_index = torch.sort(regional_attr, dim=1, descending=True)

        # 3. 预计算所有 (n, r) 组合的累积遮蔽掩码
        flat_size = channels * H * W
        row_len = W

        nr2mask = {}
        x_flat = x.view(bs, -1, H, W)  # [bs*C, H, W] -- 对于MNIST channels=1

        with torch.no_grad():
            for n in range(1, self.N + 1):
                for r in range(1, self.R + 1):
                    nr2mask[f'{n}_{r}'] = torch.zeros(bs, channels, H, W, device=device)

            for sample_idx in range(bs):
                for r in range(1, self.R + 1):
                    for i in range(self.N):
                        region_index = max_region_index[sample_idx, i]
                        # 将展平索引转换为 (channel, row, col)
                        pixel_in_chw = region_index.item()
                        selected_i = (pixel_in_chw % (H * W)) // W
                        selected_j = (pixel_in_chw % (H * W)) % W

                        left_x = max(selected_i - r, 0)
                        left_y = max(selected_j - r, 0)
                        right_x = min(selected_i + r, H)
                        right_y = min(selected_j + r, W)

                        nr2mask[f'{i+1}_{r}'][sample_idx, :, left_x:right_x, left_y:right_y] = 1

                        # 累积：第 i+1 个区域的掩码 = 第 i 个区域掩码 + 新区域
                        if i > 0:
                            nr2mask[f'{i+1}_{r}'][sample_idx] = torch.clamp(
                                nr2mask[f'{i+1}_{r}'][sample_idx] + nr2mask[f'{i}_{r}'][sample_idx],
                                0, 1
                            )

        # 4. 渐进式遮蔽 + 逐样本早停
        # 先检查哪些样本当前预测正确（只攻击预测正确的样本）
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

                    # 对仍需攻击的样本应用遮蔽
                    for idx in range(bs):
                        if not sample2perturb[idx]:
                            continue
                        occ_x[idx] = (1 - mask[idx]) * x[idx] + mask[idx] * self.c

                    occ_x = torch.clamp(occ_x, 0, 1)

                    # 检查哪些样本已被攻击成功
                    pred = self.net(occ_x).argmax(dim=1)
                    sample2perturb = (pred.cpu() == y.cpu()).numpy()

        return occ_x
