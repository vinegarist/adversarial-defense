#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""补充分析脚本：生成毕设论文新增图表

本脚本基于已有的 occlusion_attack.py 与训练好的模型，
生成以下三张可视化图：
  1. exp_fixed_vs_adaptive.pdf  —— Fixed 与 Adaptive 遮蔽对比（含重叠分析）
  2. exp_saliency_vs_ig.pdf     —— Saliency 与 IG 归因图对比（Standard vs PGD-AT）
  3. exp_transfer_anomaly.pdf   —— 白盒 vs 迁移攻击异常现象

并打印关键统计量（Fixed mask 实际遮蔽面积 vs 理论上限），
用于补充实验章节。
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import LeNet5
from occlusion_attack import (
    compute_saliency,
    SaliencyOcclusionAttack,
    AdaptiveSaliencyOcclusionAttack,
    HAS_CAPTUM,
)

if HAS_CAPTUM:
    from captum.attr import IntegratedGradients

# ============================================================
# 配置
# ============================================================
PAPER_FIG_DIR = os.path.join(ROOT, 'paper_figures')
THESIS_FIG_DIR = r'D:\软件\南开大学论文模板2026\figures'
MODEL_DIR = os.path.join(ROOT, 'save_model', '50epoch')
DATA_DIR = os.path.join(ROOT, 'data')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_K = 9
KERNEL = 3
N, R = 5, 3

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = LeNet5()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and 'net' in ckpt:
        model.load_state_dict(ckpt['net'])
    else:
        model.load_state_dict(ckpt)
    return model.to(DEVICE).eval()


def get_samples(n_samples=10, seed=42):
    """取 n_samples 个不同标签的样本，便于后续展示"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    rng = np.random.RandomState(seed)
    # 每个数字类挑一张
    by_label = {}
    indices = rng.permutation(len(testset))
    for i in indices:
        x, y = testset[int(i)]
        if y not in by_label:
            by_label[y] = (x, y)
        if len(by_label) >= n_samples:
            break
    items = sorted(by_label.items(), key=lambda kv: kv[0])[:n_samples]
    xs = torch.stack([v[0] for _, v in items])
    ys = torch.tensor([v[1] for _, v in items], dtype=torch.long)
    return xs.to(DEVICE), ys.to(DEVICE)


# ============================================================
# 工具：复刻 SaliencyOcclusionAttack 的 mask 生成（不应用遮蔽）
# 仅为可视化，复用模型梯度但只输出 mask，便于统计重叠
# ============================================================
def fixed_mask_from_saliency(model, x, y, top_k=TOP_K, kernel=KERNEL):
    """生成 Fixed 攻击的二值遮蔽 mask 与每个 top-k 块的单独 mask"""
    bs, _, H, W = x.shape
    pad = kernel // 2
    x_g = x.detach().requires_grad_(True)
    attr = compute_saliency(model, x_g, y)  # [B,1,H,W]
    attr_sum = attr.sum(dim=1, keepdim=True)
    # 等价的卷积求和（同 SaliencyOcclusionAttack）
    weight = torch.ones(1, 1, kernel, kernel, device=x.device)
    out_sum = torch.nn.functional.conv2d(attr_sum, weight, padding=pad)
    out_flat = out_sum.view(bs, -1)
    _, top_idx = torch.topk(out_flat, top_k, dim=1)

    full_mask = torch.zeros(bs, 1, H, W, device=x.device)
    per_block = torch.zeros(bs, top_k, H, W, device=x.device)
    for b in range(bs):
        for k, idx in enumerate(top_idx[b]):
            r, c = int(idx) // W, int(idx) % W
            r0, r1 = max(0, r - pad), min(H, r + pad + 1)
            c0, c1 = max(0, c - pad), min(W, c + pad + 1)
            full_mask[b, 0, r0:r1, c0:c1] = 1
            per_block[b, k, r0:r1, c0:c1] = 1
    return full_mask, per_block


def adaptive_mask_from_saliency(model, x, y, N_=N, R_=R, c_val=0.0):
    """运行 AdaptiveSaliencyOcclusionAttack 并提取所应用的 mask"""
    attacker = AdaptiveSaliencyOcclusionAttack(model, N=N_, R=R_, c=c_val)
    x_adv = attacker((x, y))
    # 对二值图像 mask = (x_adv != x)
    mask = (x_adv - x).abs() > 1e-6
    return x_adv, mask.float()


# ============================================================
# 图1：Fixed vs Adaptive 遮蔽对比（含重叠分析）
# ============================================================
def figure_fixed_vs_adaptive(model, n_samples=5):
    xs, ys = get_samples(n_samples=n_samples)

    full_mask_fix, per_block = fixed_mask_from_saliency(model, xs, ys)
    x_adv_ada, mask_ada = adaptive_mask_from_saliency(model, xs, ys)

    # ---------- 关键统计：Fixed 实际遮蔽面积 vs 理论上限 ----------
    theoretical = TOP_K * (KERNEL ** 2)  # 9 * 9 = 81
    actual_fix = full_mask_fix.view(n_samples, -1).sum(dim=1).cpu().numpy()
    overlap_loss = theoretical - actual_fix
    overlap_ratio = overlap_loss / theoretical * 100

    actual_ada = mask_ada.view(n_samples, -1).sum(dim=1).cpu().numpy()
    # 多通道 mask 已是 1×H×W 实际像素数

    print('=' * 60)
    print('Fixed mask 重叠统计（top_k=9, kernel=3, 理论上限 81 像素）：')
    for i, (a, l, r) in enumerate(zip(actual_fix, overlap_loss, overlap_ratio)):
        print(f'  样本{i}(label={int(ys[i])}): 实际={int(a):3d} 像素, 重叠损失={int(l):2d}, 重叠率={r:5.2f}%')
    print(f'  平均实际遮蔽 = {actual_fix.mean():.2f} 像素，平均重叠率 = {overlap_ratio.mean():.2f}%')
    print()
    print('Adaptive mask 实际遮蔽统计（N=5, R=3, 早停后）：')
    for i, a in enumerate(actual_ada):
        print(f'  样本{i}(label={int(ys[i])}): 实际={int(a):3d} 像素')
    print(f'  平均实际遮蔽 = {actual_ada.mean():.2f} 像素')
    print('=' * 60)

    # ---------- 绘图：4 行 × n_samples 列 ----------
    # 行1: 原图
    # 行2: Fixed 遮蔽块的"叠加可视化"（颜色编码每一块；重叠区高亮）
    # 行3: Fixed 最终 mask 应用后的图像
    # 行4: Adaptive mask 应用后的图像
    fig, axes = plt.subplots(
        4, n_samples + 1, figsize=(2.0 * n_samples + 1.4, 9.8),
        gridspec_kw={'width_ratios': [0.55] + [1] * n_samples}
    )
    row_labels = ['原图', 'Fixed叠加\npx=重叠像素',
                  f'Fixed最终\npx=遮蔽像素\n上限{theoretical}px',
                  f'Adaptive早停\npx=遮蔽像素\nN={N}, R={R}']
    for r_, lab in enumerate(row_labels):
        axes[r_, 0].axis('off')
        axes[r_, 0].text(0.5, 0.5, lab, transform=axes[r_, 0].transAxes,
                         rotation=90, va='center', ha='center', fontsize=19)
    for j in range(n_samples):
        col = j + 1
        # 原图
        img = xs[j, 0].cpu().numpy()
        axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'label={int(ys[j])}', fontsize=20)
        axes[0, col].axis('off')

        # Fixed 多块叠加（用 jet 显示每块的次序，重叠处累加 -> 颜色更亮）
        block_overlay = per_block[j].sum(dim=0).cpu().numpy()  # [H,W] 0..top_k
        rgb = np.stack([img, img, img], axis=-1)
        # 灰度底图 + 红色高亮（强度=覆盖次数）
        red_alpha = np.clip(block_overlay / block_overlay.max() if block_overlay.max() > 0 else block_overlay, 0, 1)
        rgb[..., 0] = np.clip(rgb[..., 0] + red_alpha * 0.9, 0, 1)
        rgb[..., 1] = rgb[..., 1] * (1 - red_alpha * 0.6)
        rgb[..., 2] = rgb[..., 2] * (1 - red_alpha * 0.6)
        axes[1, col].imshow(rgb)
        # 重叠像素数标注
        ov_px = int((block_overlay > 1).sum())
        axes[1, col].text(0.5, -0.045, f'{ov_px}px', transform=axes[1, col].transAxes,
                          ha='center', va='top', fontsize=22, color='#C0392B',
                          clip_on=False)
        axes[1, col].axis('off')

        # Fixed 最终遮蔽
        m = full_mask_fix[j, 0].cpu().numpy()
        x_fix = img * (1 - m)
        axes[2, col].imshow(x_fix, cmap='gray', vmin=0, vmax=1)
        axes[2, col].text(0.5, -0.045, f'{int(actual_fix[j])}px',
                          transform=axes[2, col].transAxes,
                          ha='center', va='top', fontsize=22, color='black',
                          clip_on=False)
        axes[2, col].axis('off')

        # Adaptive 最终遮蔽
        x_ad = x_adv_ada[j, 0].cpu().numpy()
        axes[3, col].imshow(x_ad, cmap='gray', vmin=0, vmax=1)
        axes[3, col].text(0.5, -0.045, f'{int(actual_ada[j])}px',
                          transform=axes[3, col].transAxes,
                          ha='center', va='top', fontsize=22, color='black',
                          clip_on=False)
        axes[3, col].axis('off')

    # 行标题
    row_labels = ['原图', 'Fixed叠加\n(越亮越重叠)',
                  f'Fixed最终\n(上限{theoretical}px)',
                  f'Adaptive早停\n(N={N},R={R})']
    for r_, lab in enumerate(row_labels):
        # Row labels are already drawn in the dedicated left column above.
        pass

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.36, wspace=0.04, bottom=0.05)
    return fig, dict(actual_fix=actual_fix.tolist(),
                     overlap_ratio=overlap_ratio.tolist(),
                     actual_ada=actual_ada.tolist(),
                     theoretical=theoretical)


# ============================================================
# 图2：Saliency vs IG 归因图对比（Standard vs PGD-AT）
# ============================================================
def figure_saliency_vs_ig(model_std, model_pgd, n_samples=4):
    xs, ys = get_samples(n_samples=n_samples)

    def saliency_attr(m, x, y):
        x_g = x.detach().requires_grad_(True)
        attr = compute_saliency(m, x_g, y)
        return attr.detach().cpu().numpy()[:, 0]  # [B,H,W]

    def ig_attr(m, x, y):
        if not HAS_CAPTUM:
            return np.zeros_like(x.cpu().numpy()[:, 0])
        x_ig = x.detach().requires_grad_(True)
        ig = IntegratedGradients(m)
        a = ig.attribute(x_ig, target=y, n_steps=50, baselines=x_ig * 0)
        return a.detach().abs().cpu().numpy()[:, 0]

    sal_std = saliency_attr(model_std, xs, ys)
    sal_pgd = saliency_attr(model_pgd, xs, ys)
    ig_std = ig_attr(model_std, xs, ys)
    ig_pgd = ig_attr(model_pgd, xs, ys)

    # 5 行 × n_samples：原图 / Sal-Std / Sal-PGD / IG-Std / IG-PGD
    fig, axes = plt.subplots(5, n_samples, figsize=(1.85 * n_samples, 10.5))

    def norm(a):
        a = np.abs(a)
        if a.max() > 0:
            a = a / a.max()
        return a

    for j in range(n_samples):
        img = xs[j, 0].cpu().numpy()
        axes[0, j].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, j].set_title(f'label={int(ys[j])}', fontsize=17)
        axes[0, j].axis('off')

        for r_, data, cmap in [
            (1, norm(sal_std[j]), 'hot'),
            (2, norm(sal_pgd[j]), 'hot'),
            (3, norm(ig_std[j]), 'viridis'),
            (4, norm(ig_pgd[j]), 'viridis'),
        ]:
            axes[r_, j].imshow(img, cmap='gray', vmin=0, vmax=1, alpha=0.35)
            axes[r_, j].imshow(data, cmap=cmap, alpha=0.75)
            axes[r_, j].axis('off')

    row_labels = ['原始样本',
                  'Saliency\n(Standard)',
                  'Saliency\n(PGD-AT)',
                  'IG (n_steps=50)\n(Standard)',
                  'IG (n_steps=50)\n(PGD-AT)']
    for r_, lab in enumerate(row_labels):
        axes[r_, 0].text(-0.22, 0.5, lab, transform=axes[r_, 0].transAxes,
                         rotation=90, va='center', ha='center', fontsize=17)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.14)

    # 量化：每个归因图的"集中度"（前 9 像素之和占总和的比例）
    def concentration(a):
        flat = np.sort(a.flatten())[::-1]
        if flat.sum() == 0:
            return 0
        return flat[:9].sum() / flat.sum()

    stats = {}
    for name, arr in [('saliency_standard', sal_std), ('saliency_pgd_at', sal_pgd),
                      ('ig_standard', ig_std), ('ig_pgd_at', ig_pgd)]:
        c = [concentration(np.abs(arr[k])) for k in range(n_samples)]
        stats[name] = float(np.mean(c))
    print('归因集中度（top-9 像素占总归因的比例，越大越集中）：')
    for k, v in stats.items():
        print(f'  {k}: {v:.4f}')
    return fig, stats


# ============================================================
# 图3：迁移攻击异常现象
# ============================================================
def figure_transfer_anomaly():
    """根据论文表 6.1 直接绘制——基于已有数据，无需重跑攻击"""
    methods = ['FGSM', 'PGD', 'C&W', '自适应遮蔽\n(N=5,R=3)']
    whitebox = [41.56, 4.63, 5.76, 94.30]
    transfer = [95.38, 93.98, 96.57, 91.96]
    gap = [t - w for t, w in zip(transfer, whitebox)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={'width_ratios': [1.6, 1]})

    x = np.arange(len(methods))
    w = 0.36
    bars1 = ax1.bar(x - w / 2, whitebox, w, label='白盒攻击', color='#E74C3C')
    bars2 = ax1.bar(x + w / 2, transfer, w, label='迁移攻击 (来自Standard)', color='#3498DB')

    for bs in (bars1, bars2):
        for b in bs:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                     f'{b.get_height():.2f}%', ha='center', fontsize=13)

    ax1.set_ylabel('防御准确率 (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=14)
    ax1.set_ylim(0, 110)
    ax1.set_title('Adaptive-Saliency-AT(N=5,R=3) 在白盒 vs 迁移攻击下的准确率')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)

    # 右图：差距柱
    colors = ['#E74C3C' if g > 50 else ('#F39C12' if g > 0 else '#27AE60') for g in gap]
    bars3 = ax2.barh(methods, gap, color=colors)
    for b, g in zip(bars3, gap):
        ax2.text(g + (1 if g >= 0 else -1), b.get_y() + b.get_height() / 2,
                 f'{g:+.2f}', va='center',
                 ha='left' if g >= 0 else 'right', fontsize=13)
    ax2.axvline(0, color='k', lw=0.8)
    ax2.set_xlabel('迁移 − 白盒 (百分点)')
    ax2.set_title('差距：>0 意味着白盒比迁移更脆弱')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)
    return fig


# ============================================================
# 主流程
# ============================================================
def save_both(fig, basename):
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    os.makedirs(THESIS_FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PAPER_FIG_DIR, f'{basename}.{ext}'))
        fig.savefig(os.path.join(THESIS_FIG_DIR, f'{basename}.{ext}'))
    plt.close(fig)


def main():
    print(f'Device: {DEVICE}')
    print(f'captum available: {HAS_CAPTUM}')

    print('\n[1/3] 生成 Fixed vs Adaptive 遮蔽对比图...')
    model_std = load_model('mnist_lenet5.pth')
    fig1, stats1 = figure_fixed_vs_adaptive(model_std, n_samples=5)
    save_both(fig1, 'exp_fixed_vs_adaptive')
    print('保存: exp_fixed_vs_adaptive.{pdf,png}')

    print('\n[2/3] 生成 Saliency vs IG 归因图对比...')
    model_pgd = load_model('mnist_lenet5_PGD_0.1_5_AT.pth')
    fig2, stats2 = figure_saliency_vs_ig(model_std, model_pgd, n_samples=4)
    save_both(fig2, 'exp_saliency_vs_ig')
    print('保存: exp_saliency_vs_ig.{pdf,png}')

    print('\n[3/3] 生成迁移攻击异常现象对比图...')
    fig3 = figure_transfer_anomaly()
    save_both(fig3, 'exp_transfer_anomaly')
    print('保存: exp_transfer_anomaly.{pdf,png}')

    # 输出统计摘要
    print('\n========== 关键统计摘要 ==========')
    print(f'Fixed mask 理论上限像素 = {stats1["theoretical"]}')
    print(f'Fixed mask 平均重叠率 = {np.mean(stats1["overlap_ratio"]):.2f}%')
    print(f'Adaptive mask 平均实际遮蔽 = {np.mean(stats1["actual_ada"]):.2f} px')
    print(f'归因集中度: {stats2}')


if __name__ == '__main__':
    main()
