#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""攻击效果对比可视化脚本.
绘制固定遮蔽攻击和自适应遮蔽攻击在标准模型上的效果对比.
包括：预测结果、准确率、参数标注.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import LeNet5
from occlusion_attack import (
    SaliencyOcclusionAttack,
    AdaptiveSaliencyOcclusionAttack,
)

# ============================================================
# 配置
# ============================================================
THESIS_FIG_DIR = os.path.join(r'D:\软件\南开大学论文模板2026\figures', 'attack_compare')
MODEL_DIR = os.path.join(ROOT, 'save_model', '50epoch')
DATA_DIR = os.path.join(ROOT, 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 150

N_SAMPLES = 5


def load_model():
    """加载标准模型."""
    model = LeNet5()
    ckpt = torch.load(os.path.join(MODEL_DIR, 'mnist_lenet5.pth'), map_location=DEVICE)
    if 'net' in ckpt:
        model.load_state_dict(ckpt['net'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    model.eval()
    return model


def get_test_samples(n=100, seed=42):
    """获取测试样本，返回normalize后用于模型输入的版本."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    testset = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transform)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(testset), n, replace=False)
    xs, ys = [], []
    for i in indices:
        x, y = testset[i]
        xs.append(x)
        ys.append(y)
    xs = torch.stack(xs).to(DEVICE)
    ys = torch.tensor(ys, dtype=torch.long).to(DEVICE)
    return xs, ys


def predict(model, x):
    """预测."""
    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)
        pred = prob.argmax(dim=1)
        conf = prob.max(dim=1)[0]
    return pred.cpu().numpy(), conf.cpu().numpy()


def run_fixed_attack_get_mask(model, x, y, top_k, kernel_size=3):
    """运行固定遮蔽攻击并获取mask."""
    attack = SaliencyOcclusionAttack(model, top_k=top_k, kernel_size=kernel_size, occlu_color=0.0)
    x_adv = attack((x.clone(), y))

    # 正确的mask检测：只标记原本是笔画区域但被遮蔽的区域
    # 原始图像中笔画区域像素值较高（normalize后约0.5-2.8）
    # 原始背景区域像素值低（normalize后约-0.424，被clamp到0）
    # 遮蔽区域：原本是笔画（值高），攻击后变低（被遮蔽）

    x_orig_np = x[0, 0].cpu().numpy()
    x_adv_np = x_adv[0, 0].cpu().numpy()

    # 原本是笔画区域（像素值>0.5，排除背景）
    is_stroke_orig = x_orig_np > 0.3  # normalize后的阈值
    # 攻击后像素值变低（被遮蔽）
    is_occluded = x_adv_np < 0.1

    # mask = 原本是笔画 AND 攻击后被遮蔽
    mask = (is_stroke_orig & is_occluded).astype(float)

    return x_adv, mask


def run_adaptive_attack_get_mask(model, x, y, N, R):
    """运行自适应遮蔽攻击并获取mask."""
    attack = AdaptiveSaliencyOcclusionAttack(model, N=N, R=R, c=0.0)
    x_adv = attack((x.clone(), y))

    x_orig_np = x[0, 0].cpu().numpy()
    x_adv_np = x_adv[0, 0].cpu().numpy()

    is_stroke_orig = x_orig_np > 0.3
    is_occluded = x_adv_np < 0.1
    mask = (is_stroke_orig & is_occluded).astype(float)

    return x_adv, mask


def title_pred(pred, conf, true_l):
    """标题格式."""
    ok = 'OK' if pred == true_l else 'X'
    return f'真:{true_l} 预:{pred}({conf:.0f}%) {ok}'


def status_color(pred, true_l):
    """状态颜色."""
    return 'green' if pred == true_l else 'red'


def overlay_red(img, mask):
    """红色叠加遮蔽区域."""
    rgb = np.stack([img, img, img], axis=-1).astype(float)
    rgb[..., 0] = np.clip(rgb[..., 0] + mask * 0.6, 0, 1)
    rgb[..., 1] *= (1 - mask * 0.4)
    rgb[..., 2] *= (1 - mask * 0.4)
    return rgb


def overlay_blue(img, mask):
    """蓝色叠加遮蔽区域."""
    rgb = np.stack([img, img, img], axis=-1).astype(float)
    rgb[..., 2] = np.clip(rgb[..., 2] + mask * 0.6, 0, 1)
    rgb[..., 0] *= (1 - mask * 0.4)
    rgb[..., 1] *= (1 - mask * 0.4)
    return rgb


# ============================================================
# 图1: Fixed-Saliency攻击效果（不同k参数）
# ============================================================
def fig_fixed_saliency(model, xs, ys):
    """固定遮蔽攻击在不同k参数下的可视化."""
    k_values = [3, 5, 9]
    kernel_size = 3

    # 选择攻击成功的样本
    success_samples = []
    for i in range(len(xs)):
        true_l = ys[i].item()
        pred_orig, _ = predict(model, xs[i:i+1])
        if pred_orig[0] == true_l:
            x_adv, mask = run_fixed_attack_get_mask(model, xs[i:i+1], ys[i:i+1], top_k=9, kernel_size=kernel_size)
            pred_adv, _ = predict(model, x_adv)
            if pred_adv[0] != true_l:
                success_samples.append(i)
        if len(success_samples) >= N_SAMPLES:
            break

    if len(success_samples) < N_SAMPLES:
        for i in range(len(xs)):
            if i not in success_samples:
                success_samples.append(i)
            if len(success_samples) >= N_SAMPLES:
                break

    fig, axes = plt.subplots(N_SAMPLES, len(k_values) + 1, figsize=(10, 2.2 * N_SAMPLES))

    for row, idx in enumerate(success_samples[:N_SAMPLES]):
        true_l = ys[idx].item()
        x_orig = xs[idx:idx+1]
        img_orig = x_orig[0, 0].cpu().numpy()

        # 原图（normalize后的值范围[-0.424, 2.821]，显示时映射到gray）
        pred_o, conf_o = predict(model, x_orig)
        axes[row, 0].imshow(img_orig, cmap='gray', vmin=-0.5, vmax=2.5)
        if row == 0:
            axes[row, 0].set_title('原图\n干净准确率99%', fontsize=9)
        axes[row, 0].text(-0.15, 0.5, title_pred(pred_o[0], conf_o[0], true_l),
                         transform=axes[row, 0].transAxes, rotation=90,
                         va='center', ha='center', fontsize=8,
                         color=status_color(pred_o[0], true_l))
        axes[row, 0].axis('off')

        # 不同k值的攻击结果
        for col, k in enumerate(k_values):
            x_adv, mask = run_fixed_attack_get_mask(model, x_orig, ys[idx:idx+1], top_k=k, kernel_size=kernel_size)
            img_adv = x_adv[0, 0].cpu().numpy()

            pred_a, conf_a = predict(model, x_adv)

            # 显示攻击后的图像（带红色遮蔽标记）
            # 由于clamp，图像值范围是[0, 1]
            rgb = overlay_red(img_adv, mask)

            axes[row, col+1].imshow(rgb)
            if row == 0:
                acc_map = {3: 86.82, 5: 80.99, 9: 71.19}
                axes[row, col+1].set_title(f'Fixed-Saliency\nk={k}, kernel={kernel_size}\n准确率{acc_map[k]}%', fontsize=9)
            axes[row, col+1].text(-0.15, 0.5, title_pred(pred_a[0], conf_a[0], true_l),
                                  transform=axes[row, col+1].transAxes, rotation=90,
                                  va='center', ha='center', fontsize=8,
                                  color=status_color(pred_a[0], true_l))
            axes[row, col+1].text(0.95, 0.05, f'{int(mask.sum())}px',
                                 transform=axes[row, col+1].transAxes,
                                 fontsize=7, ha='right', color='darkred')
            axes[row, col+1].axis('off')

    handles = [mpatches.Patch(color='red', alpha=0.5, label='遮蔽区域')]
    fig.legend(handles=handles, loc='upper right', fontsize=8)
    fig.suptitle('Fixed-Saliency遮蔽攻击效果（标准模型）\n参数说明：k为遮蔽区域数，kernel_size为遮蔽块大小', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(THESIS_FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(THESIS_FIG_DIR, 'fixed_saliency_compare.png'))
    plt.close(fig)
    print('保存: fixed_saliency_compare.png')


# ============================================================
# 图2: Adaptive-Saliency攻击效果（固定R=3，变化N）
# ============================================================
def fig_adaptive_n(model, xs, ys):
    """自适应遮蔽攻击 - 固定R变化N."""
    R_fixed = 3
    N_values = [3, 5, 10]

    success_samples = []
    for i in range(len(xs)):
        true_l = ys[i].item()
        pred_orig, _ = predict(model, xs[i:i+1])
        if pred_orig[0] == true_l:
            x_adv, mask = run_adaptive_attack_get_mask(model, xs[i:i+1], ys[i:i+1], N=10, R=R_fixed)
            pred_adv, _ = predict(model, x_adv)
            if pred_adv[0] != true_l:
                success_samples.append(i)
        if len(success_samples) >= N_SAMPLES:
            break

    if len(success_samples) < N_SAMPLES:
        for i in range(len(xs)):
            if i not in success_samples:
                success_samples.append(i)
            if len(success_samples) >= N_SAMPLES:
                break

    fig, axes = plt.subplots(N_SAMPLES, len(N_values) + 1, figsize=(10, 2.2 * N_SAMPLES))

    for row, idx in enumerate(success_samples[:N_SAMPLES]):
        true_l = ys[idx].item()
        x_orig = xs[idx:idx+1]
        img_orig = x_orig[0, 0].cpu().numpy()

        pred_o, conf_o = predict(model, x_orig)
        axes[row, 0].imshow(img_orig, cmap='gray', vmin=-0.5, vmax=2.5)
        if row == 0:
            axes[row, 0].set_title('原图\n干净准确率99%', fontsize=9)
        axes[row, 0].text(-0.15, 0.5, title_pred(pred_o[0], conf_o[0], true_l),
                         transform=axes[row, 0].transAxes, rotation=90,
                         va='center', ha='center', fontsize=8,
                         color=status_color(pred_o[0], true_l))
        axes[row, 0].axis('off')

        for col, N in enumerate(N_values):
            x_adv, mask = run_adaptive_attack_get_mask(model, x_orig, ys[idx:idx+1], N=N, R=R_fixed)
            img_adv = x_adv[0, 0].cpu().numpy()

            pred_a, conf_a = predict(model, x_adv)
            rgb = overlay_blue(img_adv, mask)

            axes[row, col+1].imshow(rgb)
            if row == 0:
                acc_map = {3: 49.45, 5: 34.33, 10: 15.95}
                axes[row, col+1].set_title(f'Adaptive-Saliency\nN={N}, R={R_fixed}\n准确率{acc_map[N]}%', fontsize=9)
            axes[row, col+1].text(-0.15, 0.5, title_pred(pred_a[0], conf_a[0], true_l),
                                  transform=axes[row, col+1].transAxes, rotation=90,
                                  va='center', ha='center', fontsize=8,
                                  color=status_color(pred_a[0], true_l))
            axes[row, col+1].text(0.95, 0.05, f'{int(mask.sum())}px',
                                 transform=axes[row, col+1].transAxes,
                                 fontsize=7, ha='right', color='darkblue')
            axes[row, col+1].axis('off')

    handles = [mpatches.Patch(color='blue', alpha=0.5, label='遮蔽区域')]
    fig.legend(handles=handles, loc='upper right', fontsize=8)
    fig.suptitle('Adaptive-Saliency遮蔽攻击效果（标准模型）\n固定R=3，变化N（迭代次数）', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(os.path.join(THESIS_FIG_DIR, 'adaptive_saliency_N_compare.png'))
    plt.close(fig)
    print('保存: adaptive_saliency_N_compare.png')


# ============================================================
# 图3: Adaptive-Saliency攻击效果（固定N=5，变化R）
# ============================================================
def fig_adaptive_r(model, xs, ys):
    """自适应遮蔽攻击 - 固定N变化R."""
    N_fixed = 5
    R_values = [2, 3, 4]

    success_samples = []
    for i in range(len(xs)):
        true_l = ys[i].item()
        pred_orig, _ = predict(model, xs[i:i+1])
        if pred_orig[0] == true_l:
            x_adv, mask = run_adaptive_attack_get_mask(model, xs[i:i+1], ys[i:i+1], N=N_fixed, R=4)
            pred_adv, _ = predict(model, x_adv)
            if pred_adv[0] != true_l:
                success_samples.append(i)
        if len(success_samples) >= N_SAMPLES:
            break

    if len(success_samples) < N_SAMPLES:
        for i in range(len(xs)):
            if i not in success_samples:
                success_samples.append(i)
            if len(success_samples) >= N_SAMPLES:
                break

    fig, axes = plt.subplots(N_SAMPLES, len(R_values) + 1, figsize=(10, 2.2 * N_SAMPLES))

    for row, idx in enumerate(success_samples[:N_SAMPLES]):
        true_l = ys[idx].item()
        x_orig = xs[idx:idx+1]
        img_orig = x_orig[0, 0].cpu().numpy()

        pred_o, conf_o = predict(model, x_orig)
        axes[row, 0].imshow(img_orig, cmap='gray', vmin=-0.5, vmax=2.5)
        if row == 0:
            axes[row, 0].set_title('原图\n干净准确率99%', fontsize=9)
        axes[row, 0].text(-0.15, 0.5, title_pred(pred_o[0], conf_o[0], true_l),
                         transform=axes[row, 0].transAxes, rotation=90,
                         va='center', ha='center', fontsize=8,
                         color=status_color(pred_o[0], true_l))
        axes[row, 0].axis('off')

        for col, R in enumerate(R_values):
            x_adv, mask = run_adaptive_attack_get_mask(model, x_orig, ys[idx:idx+1], N=N_fixed, R=R)
            img_adv = x_adv[0, 0].cpu().numpy()

            pred_a, conf_a = predict(model, x_adv)
            rgb = overlay_blue(img_adv, mask)

            kernel_size = 2 * R + 1
            axes[row, col+1].imshow(rgb)
            if row == 0:
                acc_map = {2: 49.15, 3: 34.33, 4: 23.91}
                axes[row, col+1].set_title(f'Adaptive-Saliency\nN={N_fixed}, R={R}, kernel={kernel_size}\n准确率{acc_map[R]}%', fontsize=9)
            axes[row, col+1].text(-0.15, 0.5, title_pred(pred_a[0], conf_a[0], true_l),
                                  transform=axes[row, col+1].transAxes, rotation=90,
                                  va='center', ha='center', fontsize=8,
                                  color=status_color(pred_a[0], true_l))
            axes[row, col+1].text(0.95, 0.05, f'{int(mask.sum())}px',
                                 transform=axes[row, col+1].transAxes,
                                 fontsize=7, ha='right', color='darkblue')
            axes[row, col+1].axis('off')

    handles = [mpatches.Patch(color='blue', alpha=0.5, label='遮蔽区域')]
    fig.legend(handles=handles, loc='upper right', fontsize=8)
    fig.suptitle('Adaptive-Saliency遮蔽攻击效果（标准模型）\n固定N=5，变化R（遮蔽半径，kernel_size=2R+1）', fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(os.path.join(THESIS_FIG_DIR, 'adaptive_saliency_R_compare.png'))
    plt.close(fig)
    print('保存: adaptive_saliency_R_compare.png')


# ============================================================
# 主函数
# ============================================================
def main():
    print('>>> 攻击效果对比可视化')

    model = load_model()
    xs, ys = get_test_samples(n=200, seed=42)

    fig_fixed_saliency(model, xs, ys)
    fig_adaptive_n(model, xs, ys)
    fig_adaptive_r(model, xs, ys)

    print('>>> 完成')


if __name__ == '__main__':
    main()