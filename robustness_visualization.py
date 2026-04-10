"""
15.4 鲁棒性评测可视化代码
为每个攻击类型创建可视化，显示0-9每个数字的攻击效果
"""

import sys
sys.path.insert(0, r'D:\软件\对抗性防御\对抗性防御-1\03.代码')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

from occlusion_attack import SaliencyOcclusionAttack, AdaptiveSaliencyOcclusionAttack
from pgd import LinfPGD
from loss import CWLoss
from models import LeNet5
from utils import load_mnist_test
import test
test_fn = test.test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print(f'Device: {device}')

# ========== 1. 加载模型 ==========
def load_model_from_ckpt(ckpt_path):
    """从检查点加载模型"""
    if not os.path.exists(ckpt_path):
        print(f"警告: 模型文件不存在 {ckpt_path}")
        return None
    net = LeNet5().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state['net'])
    net.eval()
    print(f"已加载模型: {ckpt_path}")
    return net

# 加载标准模型
std_lenet = load_model_from_ckpt('./save_model/50epoch/mnist_lenet5.pth')

# 加载Adaptive-Saliency-AT模型 (N=5, R=3)
adaptive_at_model = load_model_from_ckpt('./save_model/50epoch/mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth')

# 加载Mix-AT模型
mix_at_model = load_model_from_ckpt('./save_model/10epoch/mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth')

# 加载测试数据
imgs, lbls = load_mnist_test()

# ========== 2. 辅助函数 ==========
def imshow_with_pred(img, model, true_label, ax=None, title_prefix=''):
    """显示图像并标注真实标签和预测标签"""
    npimg = img.cpu().squeeze().numpy()
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device)).argmax(dim=1).item()

    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    ax.imshow(npimg, cmap='gray')
    correct_str = 'OK' if pred == true_label else 'X'
    ax.set_title(f'{title_prefix}\n真实:{true_label} 预测:{pred} {correct_str}', fontsize=10)
    ax.axis('off')
    return pred == true_label

# 选择展示样本：每个数字一个样本
sample_indices = []
shown = set()
for i in range(len(lbls)):
    label = int(lbls[i].item())
    if label not in shown:
        shown.add(label)
        sample_indices.append(i)
    if len(shown) == 10:
        break

print(f'展示样本索引: {sample_indices}')
print(f'样本标签: {[int(lbls[i].item()) for i in sample_indices]}')

# 创建结果目录
os.makedirs('./results_figures', exist_ok=True)

# ========== 3. 固定遮蔽攻击可视化 (不同top_k) ==========
def visualize_fixed_attack(model, model_name, top_k_values=[3, 5, 9, 15], kernel_size=3):
    """可视化固定遮蔽攻击效果"""
    fig, axes = plt.subplots(10, len(top_k_values)+1, figsize=(3*(len(top_k_values)+1), 30))

    for row, idx in enumerate(sample_indices):
        x = imgs[idx:idx+1].to(device)
        y = lbls[idx:idx+1].to(device)
        true_label = int(y.item())

        # 干净样本
        imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

        # 不同top_k的固定遮蔽攻击
        for col, top_k in enumerate(top_k_values):
            attack = SaliencyOcclusionAttack(model, top_k=top_k, kernel_size=kernel_size, occlu_color=0.0)
            x_adv = attack((x, y))
            imshow_with_pred(x_adv.squeeze(0), model, true_label,
                            ax=axes[row, col+1], title_prefix=f'Fixed k={top_k}')

    plt.tight_layout()
    filename = f'./results_figures/saliency_fixed_attack_visualization_{model_name}_ks{kernel_size}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'固定遮蔽攻击可视化已保存: {filename}')
    return fig

# ========== 4. 自适应遮蔽攻击可视化 (不同N) ==========
def visualize_adaptive_attack_n(model, model_name, N_values=[3, 5, 9, 15], R=3):
    """可视化自适应遮蔽攻击效果 (不同N参数)"""
    fig, axes = plt.subplots(10, len(N_values)+1, figsize=(3*(len(N_values)+1), 30))

    for row, idx in enumerate(sample_indices):
        x = imgs[idx:idx+1].to(device)
        y = lbls[idx:idx+1].to(device)
        true_label = int(y.item())

        # 干净样本
        imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

        # 不同N参数的自适应遮蔽攻击
        for col, N_val in enumerate(N_values):
            attack = AdaptiveSaliencyOcclusionAttack(model, N=N_val, R=R, c=0.0)
            x_adv = attack((x, y))
            imshow_with_pred(x_adv.squeeze(0), model, true_label,
                            ax=axes[row, col+1], title_prefix=f'Adaptive N={N_val}')

    plt.tight_layout()
    filename = f'./results_figures/saliency_adaptive_attack_N_visualization_{model_name}_R{R}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'自适应遮蔽攻击(N参数)可视化已保存: {filename}')
    return fig

# ========== 5. 自适应遮蔽攻击可视化 (不同R) ==========
def visualize_adaptive_attack_r(model, model_name, N=5, R_values=[1, 2, 3, 4]):
    """可视化自适应遮蔽攻击效果 (不同R参数)"""
    fig, axes = plt.subplots(10, len(R_values)+1, figsize=(3*(len(R_values)+1), 30))

    for row, idx in enumerate(sample_indices):
        x = imgs[idx:idx+1].to(device)
        y = lbls[idx:idx+1].to(device)
        true_label = int(y.item())

        # 干净样本
        imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

        # 不同R参数的自适应遮蔽攻击
        for col, R_val in enumerate(R_values):
            attack = AdaptiveSaliencyOcclusionAttack(model, N=N, R=R_val, c=0.0)
            x_adv = attack((x, y))
            imshow_with_pred(x_adv.squeeze(0), model, true_label,
                            ax=axes[row, col+1], title_prefix=f'Adaptive R={R_val}')

    plt.tight_layout()
    filename = f'./results_figures/saliency_adaptive_attack_R_visualization_{model_name}_N{N}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'自适应遮蔽攻击(R参数)可视化已保存: {filename}')
    return fig

# ========== 6. 不同遮蔽颜色可视化 ==========
def visualize_occlusion_color(model, model_name, colors=[0.0, 0.5, 1.0], N=5, R=3):
    """可视化不同遮蔽颜色的攻击效果"""
    color_names = ['黑色', '灰色', '白色']
    fig, axes = plt.subplots(10, len(colors)+1, figsize=(3*(len(colors)+1), 30))

    for row, idx in enumerate(sample_indices):
        x = imgs[idx:idx+1].to(device)
        y = lbls[idx:idx+1].to(device)
        true_label = int(y.item())

        # 干净样本
        imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

        # 不同颜色的自适应遮蔽攻击
        for col, (c, name) in enumerate(zip(colors, color_names)):
            attack = AdaptiveSaliencyOcclusionAttack(model, N=N, R=R, c=c)
            x_adv = attack((x, y))
            imshow_with_pred(x_adv.squeeze(0), model, true_label,
                            ax=axes[row, col+1], title_prefix=f'颜色={name}')

    plt.tight_layout()
    filename = f'./results_figures/saliency_color_attack_visualization_{model_name}_N{N}_R{R}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'遮蔽颜色攻击可视化已保存: {filename}')
    return fig

# ========== 7. PGD攻击可视化 ==========
def visualize_pgd_attack(model, model_name, eps_values=[0.05, 0.1, 0.15, 0.2], steps=20):
    """可视化PGD攻击效果"""
    fig, axes = plt.subplots(10, len(eps_values)+1, figsize=(3*(len(eps_values)+1), 30))

    for row, idx in enumerate(sample_indices):
        x = imgs[idx:idx+1].to(device)
        y = lbls[idx:idx+1].to(device)
        true_label = int(y.item())

        # 干净样本
        imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

        # 不同epsilon的PGD攻击
        for col, eps in enumerate(eps_values):
            pgd = LinfPGD(net=model, eps=eps, step=steps, step_size=eps/steps*2, random_start=True)
            x_adv = pgd((x, y))
            imshow_with_pred(x_adv.squeeze(0), model, true_label,
                            ax=axes[row, col+1], title_prefix=f'PGD ε={eps}')

    plt.tight_layout()
    filename = f'./results_figures/pgd_attack_visualization_{model_name}_steps{steps}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'PGD攻击可视化已保存: {filename}')
    return fig

# ========== 8. 综合对比可视化 ==========
def visualize_all_attacks_comparison(model, model_name):
    """综合对比所有攻击类型"""
    fig, axes = plt.subplots(10, 6, figsize=(18, 30))

    # 定义攻击参数
    fixed_k = 9
    adaptive_n = 5
    adaptive_r = 3
    pgd_eps = 0.1

    for row, idx in enumerate(sample_indices):
        x = imgs[idx:idx+1].to(device)
        y = lbls[idx:idx+1].to(device)
        true_label = int(y.item())

        # 1. 干净样本
        imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

        # 2. 固定遮蔽攻击
        fixed_attack = SaliencyOcclusionAttack(model, top_k=fixed_k, kernel_size=3, occlu_color=0.0)
        x_adv_fixed = fixed_attack((x, y))
        imshow_with_pred(x_adv_fixed.squeeze(0), model, true_label,
                        ax=axes[row, 1], title_prefix=f'Fixed k={fixed_k}')

        # 3. 自适应遮蔽攻击
        adaptive_attack = AdaptiveSaliencyOcclusionAttack(model, N=adaptive_n, R=adaptive_r, c=0.0)
        x_adv_adaptive = adaptive_attack((x, y))
        imshow_with_pred(x_adv_adaptive.squeeze(0), model, true_label,
                        ax=axes[row, 2], title_prefix=f'Adaptive N={adaptive_n},R={adaptive_r}')

        # 4. PGD攻击
        pgd = LinfPGD(net=model, eps=pgd_eps, step=20, step_size=0.025, random_start=True)
        x_adv_pgd = pgd((x, y))
        imshow_with_pred(x_adv_pgd.squeeze(0), model, true_label,
                        ax=axes[row, 3], title_prefix=f'PGD ε={pgd_eps}')

        # 5. FGSM攻击
        fgsm = LinfPGD(net=model, eps=pgd_eps, step=1, step_size=pgd_eps, random_start=False)
        x_adv_fgsm = fgsm((x, y))
        imshow_with_pred(x_adv_fgsm.squeeze(0), model, true_label,
                        ax=axes[row, 4], title_prefix=f'FGSM ε={pgd_eps}')

        # 6. CW攻击
        cw = LinfPGD(net=model, eps=pgd_eps, step=20, step_size=0.025, random_start=True, criterion=CWLoss)
        x_adv_cw = cw((x, y))
        imshow_with_pred(x_adv_cw.squeeze(0), model, true_label,
                        ax=axes[row, 5], title_prefix='CW')

    plt.tight_layout()
    filename = f'./results_figures/all_attacks_comparison_{model_name}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'综合攻击对比可视化已保存: {filename}')
    return fig


# ========== 主执行部分 ==========
if __name__ == '__main__':
    # 对标准模型进行可视化
    if std_lenet is not None:
        print("\n" + "="*60)
        print("对标准模型进行可视化")
        print("="*60)

        # 固定遮蔽攻击 (不同top_k)
        visualize_fixed_attack(std_lenet, 'Standard', top_k_values=[3, 5, 9, 15], kernel_size=3)

        # 自适应遮蔽攻击 (不同N)
        visualize_adaptive_attack_n(std_lenet, 'Standard', N_values=[3, 5, 9, 15], R=3)

        # 自适应遮蔽攻击 (不同R)
        visualize_adaptive_attack_r(std_lenet, 'Standard', N=5, R_values=[1, 2, 3, 4])

        # 不同遮蔽颜色
        visualize_occlusion_color(std_lenet, 'Standard', colors=[0.0, 0.5, 1.0], N=5, R=3)

        # PGD攻击
        visualize_pgd_attack(std_lenet, 'Standard', eps_values=[0.05, 0.1, 0.15, 0.2], steps=20)

        # 综合对比
        visualize_all_attacks_comparison(std_lenet, 'Standard')

    # 对Adaptive-Saliency-AT模型进行可视化
    if adaptive_at_model is not None:
        print("\n" + "="*60)
        print("对Adaptive-Saliency-AT模型进行可视化")
        print("="*60)

        visualize_fixed_attack(adaptive_at_model, 'AdaptiveSaliencyAT', top_k_values=[3, 5, 9, 15], kernel_size=3)
        visualize_adaptive_attack_n(adaptive_at_model, 'AdaptiveSaliencyAT', N_values=[3, 5, 9, 15], R=3)
        visualize_adaptive_attack_r(adaptive_at_model, 'AdaptiveSaliencyAT', N=5, R_values=[1, 2, 3, 4])
        visualize_all_attacks_comparison(adaptive_at_model, 'AdaptiveSaliencyAT')

    # 对Mix-AT模型进行可视化
    if mix_at_model is not None:
        print("\n" + "="*60)
        print("对Mix-AT模型进行可视化")
        print("="*60)

        visualize_fixed_attack(mix_at_model, 'MixAT', top_k_values=[3, 5, 9, 15], kernel_size=3)
        visualize_adaptive_attack_n(mix_at_model, 'MixAT', N_values=[3, 5, 9, 15], R=3)
        visualize_adaptive_attack_r(mix_at_model, 'MixAT', N=5, R_values=[1, 2, 3, 4])
        visualize_all_attacks_comparison(mix_at_model, 'MixAT')

    print("\n" + "="*60)
    print("所有可视化完成!")
    print("="*60)
