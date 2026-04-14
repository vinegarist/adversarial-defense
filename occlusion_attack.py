import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# captum是可选依赖，仅IG-based攻击需要
try:
    from captum.attr import IntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


def compute_saliency(model, x, y):
    """计算简单梯度归因图（与参考项目myGrad.saliency一致）

    参数:
        model: 神经网络模型
        x: 输入图像 [B, C, H, W]，需要requires_grad
        y: 标签 [B,]

    返回:
        saliency_map: 梯度归因图 [B, C, H, W]
    """
    model.eval()
    output = model(x)
    scores = output.gather(1, y.view(-1, 1))  # 取正确类别的分数
    saliency_map = torch.autograd.grad(scores, x, torch.ones_like(scores))[0]
    return saliency_map.abs()


class SaliencyOcclusionAttack(nn.Module):
    """基于Saliency（简单梯度）的固定遮蔽攻击

    与参考项目的遮蔽攻击逻辑一致，使用简单梯度而非IntegratedGradients。
    """

    def __init__(self, net, top_k=9, occlu_color=0.0, kernel_size=3):
        """
        参数说明：
        net: 待攻击的模型
        top_k: 选择梯度值最大的前top_k个像素作为遮蔽中心
        occlu_color: 遮蔽颜色，0为黑色，1为白色，0.5为灰色
        kernel_size: 遮蔽窗口大小
        """
        super(SaliencyOcclusionAttack, self).__init__()
        self.net = net
        self.top_k = top_k
        self.occlu_color = occlu_color
        self.kernel_size = kernel_size

    def forward(self, inputs):
        x, y = inputs
        device = x.device
        bs = x.shape[0]
        channels = x.shape[1]
        H, W = x.shape[2], x.shape[3]
        padding = self.kernel_size // 2

        # 使用简单梯度计算saliency
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            attr_map = compute_saliency(self.net, x_grad, y)

        # 对多通道求和（与参考项目一致）
        if channels > 1:
            attr_sum = attr_map.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        else:
            attr_sum = attr_map

        # 使用卷积计算每个位置的kernel_size邻域的梯度总和
        conv_sum = nn.Conv2d(1, 1, kernel_size=self.kernel_size,
                             stride=1, padding=padding, bias=False).to(device)
        conv_sum.weight = nn.Parameter(
            torch.ones([1, 1, self.kernel_size, self.kernel_size],
                       dtype=torch.float32).to(device))
        conv_sum.weight.requires_grad_(False)

        out_sum = conv_sum(attr_sum)  # [B, 1, H, W]

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


class AdaptiveSaliencyOcclusionAttack(nn.Module):
    """基于Saliency（简单梯度）的自适应遮蔽攻击

    与参考项目的inductiveOcclusionAttack一致：
    - 使用简单梯度计算归因图
    - 渐进式遮蔽 n=1..N, r=1..R
    - 逐样本早停
    """

    def __init__(self, net, N=5, R=3, c=0.0):
        """
        参数说明：
        net: 待攻击的模型
        N: 最大遮蔽区域数量
        R: 最大遮蔽半径
        c: 遮蔽颜色值
        """
        super(AdaptiveSaliencyOcclusionAttack, self).__init__()
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

        # 1. 使用简单梯度计算归因图
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            attr_map = compute_saliency(self.net, x_grad, y)

        # 2. 对多通道求和后展平排序
        if channels > 1:
            attr_flat = attr_map.sum(dim=1).view(bs, -1)  # [B, H*W]
        else:
            attr_flat = attr_map.view(bs, -1)

        _, max_region_index = torch.sort(attr_flat, dim=1, descending=True)

        # 3. 预计算所有 (n, r) 组合的累积遮蔽掩码
        nr2mask = {}
        with torch.no_grad():
            for n in range(1, self.N + 1):
                for r in range(1, self.R + 1):
                    nr2mask[f'{n}_{r}'] = torch.zeros(bs, channels, H, W, device=device)

            for sample_idx in range(bs):
                for r in range(1, self.R + 1):
                    for i in range(self.N):
                        region_index = max_region_index[sample_idx, i]
                        # 展平索引转换为 (row, col)
                        selected_i = region_index.item() // W
                        selected_j = region_index.item() % W

                        left_x = max(selected_i - r, 0)
                        left_y = max(selected_j - r, 0)
                        right_x = min(selected_i + r + 1, H)
                        right_y = min(selected_j + r + 1, W)

                        nr2mask[f'{i+1}_{r}'][sample_idx, :, left_x:right_x, left_y:right_y] = 1

                        # 累积
                        if i > 0:
                            nr2mask[f'{i+1}_{r}'][sample_idx] = torch.clamp(
                                nr2mask[f'{i+1}_{r}'][sample_idx] + nr2mask[f'{i}_{r}'][sample_idx],
                                0, 1
                            )

        # 4. 渐进式遮蔽 + 逐样本早停
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


class OcclusionAttack(nn.Module):
    """基于IntegratedGradients的固定遮蔽攻击（需要captum库）"""
    def __init__(self, net, top_k=9, occlu_color=0.0, kernel_size=3):
        """
        参数说明：
        net: 待攻击的模型
        top_k: 选择梯度积分值最大的前top_k个像素作为遮蔽中心
        occlu_color: 遮蔽颜色，0为黑色，1为白色，0.5为灰色
        kernel_size: 遮蔽窗口大小
        """
        super(OcclusionAttack, self).__init__()
        if not HAS_CAPTUM:
            raise ImportError("OcclusionAttack需要captum库，请使用SaliencyOcclusionAttack替代")
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
    """基于IntegratedGradients的自适应遮蔽攻击（需要captum库）

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
        if not HAS_CAPTUM:
            raise ImportError("AdaptiveOcclusionAttack需要captum库，请使用AdaptiveSaliencyOcclusionAttack替代")
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


# ============================================================
# 以下为迁移自 inequality 项目的 IG-based 遮蔽攻击
# 使用 captum IntegratedGradients 计算归因图，
# 采用 inequality 项目的 get_feature_map_avg + get_perturb_mask 逻辑
# ============================================================


class IGFixedOcclusionAttack(nn.Module):
    """基于IntegratedGradients的固定遮蔽攻击（迁移自inequality项目）

    使用captum的IntegratedGradients计算归因图（n_steps=50），
    通道均值降维后按归因值排序，选择top_k个最重要像素，
    用半径r的窗口进行遮蔽。遮蔽掩码生成逻辑与inequality项目的
    OcclusionAttack.get_perturb_mask完全一致。
    """

    def __init__(self, net, top_k=9, occlu_color=0.0, kernel_size=3, n_steps=50):
        """
        参数说明：
        net: 待攻击的模型
        top_k: 选择归因值最大的前top_k个像素作为遮蔽中心
        occlu_color: 遮蔽颜色，0为黑色，1为白色，0.5为灰色
        kernel_size: 遮蔽窗口大小（实际半径 r = kernel_size // 2）
        n_steps: IntegratedGradients积分步数（默认50，与inequality一致）
        """
        super(IGFixedOcclusionAttack, self).__init__()
        if not HAS_CAPTUM:
            raise ImportError("IGFixedOcclusionAttack需要captum库，请安装: pip install captum")
        self.net = net
        self.top_k = top_k
        self.occlu_color = occlu_color
        self.r = kernel_size // 2
        self.n_steps = n_steps

    def forward(self, inputs):
        x, y = inputs
        device = x.device
        bs = x.shape[0]
        channels = x.shape[1]
        H, W = x.shape[2], x.shape[3]

        # 1. 计算IntegratedGradients归因图（与inequality项目get_gradshap一致）
        with torch.enable_grad():
            x_ig = x.detach().requires_grad_()
            ig = IntegratedGradients(self.net)
            attr_ig = ig.attribute(
                x_ig, target=y, n_steps=self.n_steps,
                baselines=x_ig * 0
            ).detach().float()

        # 2. 通道均值得到2D归因图（与inequality的get_feature_map_avg一致）
        attr_2d = attr_ig.mean(dim=1).cpu().numpy()  # [B, H, W]

        # 3. 对每个样本生成遮蔽（与inequality的get_perturb_mask逻辑一致）
        x_adv = x.clone().detach()
        r = self.r

        for i in range(bs):
            img_grad_np = attr_2d[i]
            img_grad_flatten = np.sort(img_grad_np.flatten())

            # 取top_k个最大归因值对应的像素
            perturb_values = [img_grad_flatten[-(j + 1)] for j in range(self.top_k)]

            mask = torch.zeros(channels, H, W, device=device)
            for value in perturb_values:
                positions = np.argwhere(img_grad_np == value)
                if len(positions) > 0:
                    px, py = positions[0]
                    mask[:, max(0, px - r):min(H, px + r), max(0, py - r):min(W, py + r)] = 1

            x_adv[i] = x[i] * (1 - mask) + self.occlu_color * mask

        return torch.clamp(x_adv, 0, 1)


class AdaptiveIGOcclusionAttack(nn.Module):
    """基于IntegratedGradients的自适应遮蔽攻击（迁移自inequality项目）

    使用captum的IntegratedGradients计算归因图，
    渐进式遮蔽：遍历(r, n)参数组合，从小到大增加遮蔽区域和半径，
    对每个样本独立判断：一旦攻击成功（预测改变）即停止。
    迭代顺序与inequality项目的OcclusionAttack.occlusion一致：
    先遍历半径r，再遍历遮蔽数量n。
    """

    def __init__(self, net, N=5, R=3, c=0.0, n_steps=50):
        """
        参数说明：
        net: 待攻击的模型
        N: 最大遮蔽区域数量
        R: 最大遮蔽半径
        c: 遮蔽颜色值（0为黑色）
        n_steps: IntegratedGradients积分步数（默认50）
        """
        super(AdaptiveIGOcclusionAttack, self).__init__()
        if not HAS_CAPTUM:
            raise ImportError("AdaptiveIGOcclusionAttack需要captum库，请安装: pip install captum")
        self.net = net
        self.N = N
        self.R = R
        self.c = c
        self.n_steps = n_steps

        # 构建参数列表（与inequality的OcclusionAttack.__init__一致：先r后c）
        self.occ_params_list = []
        for r in range(1, R + 1):
            for n in range(1, N + 1):
                self.occ_params_list.append((r, n))

    def forward(self, inputs):
        x, y = inputs
        device = x.device
        bs = x.shape[0]
        channels = x.shape[1]
        H, W = x.shape[2], x.shape[3]

        # 1. 计算IntegratedGradients归因图
        with torch.enable_grad():
            x_ig = x.detach().requires_grad_()
            ig = IntegratedGradients(self.net)
            attr_ig = ig.attribute(
                x_ig, target=y, n_steps=self.n_steps,
                baselines=x_ig * 0
            ).detach().float()

        # 2. 通道均值得到2D归因图
        attr_2d = attr_ig.mean(dim=1).cpu().numpy()  # [B, H, W]

        # 3. 预排序每个样本的归因值（避免重复计算）
        sorted_positions = []
        for i in range(bs):
            img_grad_np = attr_2d[i]
            flat = img_grad_np.flatten()
            top_indices = np.argsort(flat)[::-1][:self.N]
            positions = [np.unravel_index(idx, (H, W)) for idx in top_indices]
            sorted_positions.append(positions)

        # 4. 初始化
        occ_x = x.clone().detach()
        with torch.no_grad():
            pred_init = self.net(x).argmax(dim=1)
        sample_active = (pred_init.cpu() == y.cpu()).numpy()

        # 5. 渐进式遮蔽 + 逐样本早停
        with torch.no_grad():
            for (r, n) in self.occ_params_list:
                if sample_active.sum() == 0:
                    break

                for idx in range(bs):
                    if not sample_active[idx]:
                        continue

                    # 生成遮蔽掩码（使用前n个最重要像素，半径r）
                    mask = torch.zeros(channels, H, W, device=device)
                    for j in range(n):
                        px, py = sorted_positions[idx][j]
                        mask[:, max(0, px - r):min(H, px + r),
                             max(0, py - r):min(W, py + r)] = 1

                    occ_x[idx] = x[idx] * (1 - mask) + self.c * mask

                occ_x = torch.clamp(occ_x, 0, 1)

                # 检查攻击是否成功
                pred = self.net(occ_x).argmax(dim=1)
                sample_active = (pred.cpu() == y.cpu()).numpy()

        return occ_x
