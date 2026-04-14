"""
========== 遮蔽攻击折线图可视化（完全独立版本） ==========
内核重启后可直接运行，不依赖前面cell的执行状态
"""

import sys
sys.path.insert(0, r'D:\软件\对抗性防御\对抗性防御-1\03.代码')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from collections import defaultdict

from occlusion_attack import SaliencyOcclusionAttack, AdaptiveSaliencyOcclusionAttack
from models import LeNet5
from utils import load_mnist_test
import test
test_fn = test.test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print(f'Device: {device}')


def load_model_from_ckpt(ckpt_path):
    """从检查点加载模型"""
    if not os.path.exists(ckpt_path):
        print(f"[WARN] 模型不存在: {ckpt_path}")
        return None
    net = LeNet5().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state['net'])
    net.eval()
    print(f"[OK] 加载模型: {os.path.basename(ckpt_path)}")
    return net


def run_all_occlusion_tests(model, model_name, imgs, lbls):
    """
    运行所有遮蔽攻击测试，返回结果字典

    测试参数（与15.4节完全一致）:
    - Fixed: top_k = [3, 5, 7, 9, 12, 15], kernel_size=3
    - Adaptive N: N = [3, 5, 7, 10], R=3
    - Adaptive R: R = [2, 3, 4], N=5
    """
    results = {'model_name': model_name}

    # 1. 干净样本准确率
    clean_acc, _ = test_fn(model, imgs, lbls, bs=250, mode='clean')
    results['Clean'] = clean_acc
    print(f"  {model_name} - Clean: {clean_acc:.2f}%")

    # 2. Fixed遮蔽攻击 - 不同top_k (固定kernel_size=3)
    print(f"  {model_name} - Fixed Saliency测试...")
    top_k_values = [3, 5, 7, 9, 12, 15]
    for top_k in top_k_values:
        attack = SaliencyOcclusionAttack(model, top_k=top_k, kernel_size=3, occlu_color=0.0)
        acc, _ = test_fn(nn.Sequential(attack, model), imgs, lbls, bs=250, mode='attack')
        results[f'Fixed_k{top_k}'] = acc
        print(f"    k={top_k}: {acc:.2f}%")

    # 3. Adaptive遮蔽攻击 - 不同N (固定R=3)
    print(f"  {model_name} - Adaptive Saliency (N变化)测试...")
    N_values = [3, 5, 7, 10]
    R_fixed = 3
    for N in N_values:
        attack = AdaptiveSaliencyOcclusionAttack(model, N=N, R=R_fixed, c=0.0)
        acc, _ = test_fn(nn.Sequential(attack, model), imgs, lbls, bs=250, mode='attack')
        results[f'Adaptive_N{N}_R{R_fixed}'] = acc
        print(f"    N={N},R={R_fixed}: {acc:.2f}%")

    # 4. Adaptive遮蔽攻击 - 不同R (固定N=5)
    print(f"  {model_name} - Adaptive Saliency (R变化)测试...")
    N_fixed = 5
    R_values = [2, 3, 4]
    for R in R_values:
        attack = AdaptiveSaliencyOcclusionAttack(model, N=N_fixed, R=R, c=0.0)
        acc, _ = test_fn(nn.Sequential(attack, model), imgs, lbls, bs=250, mode='attack')
        results[f'Adaptive_N{N_fixed}_R{R}'] = acc
        print(f"    N={N_fixed},R={R}: {acc:.2f}%")

    return results


def plot_occlusion_line_charts(all_results, save_dir='./results_figures'):
    """
    绘制折线图 - 使用实际测试的所有参数

    绘制4张图:
    1. Fixed-Saliency: k=[3,5,7,9,12,15] vs Accuracy
    2. Adaptive-Saliency: N=[3,5,7,10] vs Accuracy (固定R=3)
    3. Adaptive-Saliency: R=[2,3,4] vs Accuracy (固定N=5)
    4. 综合对比: 所有攻击类型的影响
    """
    os.makedirs(save_dir, exist_ok=True)

    models = [r['model_name'] for r in all_results]

    # 颜色配置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']

    # ========== 图1: Fixed-Saliency - k值影响 ==========
    fig, ax = plt.subplots(figsize=(12, 6))

    k_values = [3, 5, 7, 9, 12, 15]

    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Fixed_k{k}', 0) for k in k_values]
        ax.plot(k_values, acc_values,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2.5, markersize=10,
                label=result['model_name'])

    # 添加Clean基线
    for idx, result in enumerate(all_results):
        ax.axhline(y=result['Clean'], color=colors[idx % len(colors)],
                   linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('k (遮蔽区域数量)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Fixed-Saliency攻击: k值影响 (kernel_size=3)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    ax.set_xticks(k_values)

    plt.tight_layout()
    save_path = f'{save_dir}/line_fixed_saliency_k_complete.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[SAVED] {save_path}')

    # 保存CSV
    csv_data = []
    for result in all_results:
        for k in k_values:
            csv_data.append({
                'Model': result['model_name'],
                'Attack_Type': 'Fixed-Saliency',
                'k': k,
                'kernel_size': 3,
                'Accuracy': result.get(f'Fixed_k{k}', 0)
            })
    pd.DataFrame(csv_data).to_csv(f'{save_dir}/data_fixed_saliency_k_complete.csv', index=False)

    # ========== 图2: Adaptive-Saliency - N值影响 (R=3) ==========
    fig, ax = plt.subplots(figsize=(12, 6))

    N_values = [3, 5, 7, 10]
    R_fixed = 3

    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Adaptive_N{N}_R{R_fixed}', 0) for N in N_values]
        ax.plot(N_values, acc_values,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2.5, markersize=10,
                label=result['model_name'])

    ax.set_xlabel('N (最大遮蔽区域数)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Adaptive-Saliency攻击: N值影响 (R={R_fixed})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    ax.set_xticks(N_values)

    plt.tight_layout()
    save_path = f'{save_dir}/line_adaptive_saliency_N_complete.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[SAVED] {save_path}')

    # 保存CSV
    csv_data = []
    for result in all_results:
        for N in N_values:
            csv_data.append({
                'Model': result['model_name'],
                'Attack_Type': 'Adaptive-Saliency',
                'N': N,
                'R': R_fixed,
                'Accuracy': result.get(f'Adaptive_N{N}_R{R_fixed}', 0)
            })
    pd.DataFrame(csv_data).to_csv(f'{save_dir}/data_adaptive_saliency_N_complete.csv', index=False)

    # ========== 图3: Adaptive-Saliency - R值影响 (N=5) ==========
    fig, ax = plt.subplots(figsize=(12, 6))

    N_fixed = 5
    R_values = [2, 3, 4]

    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Adaptive_N{N_fixed}_R{R}', 0) for R in R_values]
        ax.plot(R_values, acc_values,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2.5, markersize=10,
                label=result['model_name'])

    ax.set_xlabel('R (遮蔽半径)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Adaptive-Saliency攻击: R值影响 (N={N_fixed})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    ax.set_xticks(R_values)

    plt.tight_layout()
    save_path = f'{save_dir}/line_adaptive_saliency_R_complete.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[SAVED] {save_path}')

    # 保存CSV
    csv_data = []
    for result in all_results:
        for R in R_values:
            csv_data.append({
                'Model': result['model_name'],
                'Attack_Type': 'Adaptive-Saliency',
                'N': N_fixed,
                'R': R,
                'Accuracy': result.get(f'Adaptive_N{N_fixed}_R{R}', 0)
            })
    pd.DataFrame(csv_data).to_csv(f'{save_dir}/data_adaptive_saliency_R_complete.csv', index=False)

    # ========== 图4: 综合对比 - 所有攻击类型 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图1: Fixed k
    k_values = [3, 5, 7, 9, 12, 15]
    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Fixed_k{k}', 0) for k in k_values]
        axes[0].plot(k_values, acc_values,
                     marker=markers[idx % len(markers)],
                     color=colors[idx % len(colors)],
                     linewidth=2, markersize=8,
                     label=result['model_name'])
    axes[0].set_xlabel('k (遮蔽区域数量)', fontsize=12)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Fixed-Saliency: k值影响', fontsize=14)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)

    # 子图2: Adaptive N
    N_values = [3, 5, 7, 10]
    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Adaptive_N{N}_R3', 0) for N in N_values]
        axes[1].plot(N_values, acc_values,
                     marker=markers[idx % len(markers)],
                     color=colors[idx % len(colors)],
                     linewidth=2, markersize=8,
                     label=result['model_name'])
    axes[1].set_xlabel('N (最大遮蔽区域数)', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Adaptive-Saliency: N值影响 (R=3)', fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)

    # 子图3: Adaptive R
    R_values = [2, 3, 4]
    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Adaptive_N5_R{R}', 0) for R in R_values]
        axes[2].plot(R_values, acc_values,
                     marker=markers[idx % len(markers)],
                     color=colors[idx % len(colors)],
                     linewidth=2, markersize=8,
                     label=result['model_name'])
    axes[2].set_xlabel('R (遮蔽半径)', fontsize=12)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_title('Adaptive-Saliency: R值影响 (N=5)', fontsize=14)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 105)

    plt.tight_layout()
    save_path = f'{save_dir}/line_all_occlusion_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[SAVED] {save_path}')

    print('\n========== 所有折线图绘制完成 ==========')


# ========== 主执行 ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("遮蔽攻击折线图可视化 - 完全独立版本")
    print("="*60)

    # 1. 加载测试数据
    print("\n[1/4] 加载测试数据...")
    imgs, lbls = load_mnist_test()
    print(f"测试集大小: {len(imgs)}")

    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    models_to_test = []

    # 标准模型
    std_model = load_model_from_ckpt('./save_model/50epoch/mnist_lenet5.pth')
    if std_model:
        models_to_test.append((std_model, 'Standard'))

    # Adaptive-Saliency-AT模型
    cnn = load_model_from_ckpt('./save_model/50epoch/mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth')
    if cnn:
        models_to_test.append((cnn, 'Adaptive-Saliency-AT'))

    # Mix-AT模型 (如果存在)
    cnn_mix = load_model_from_ckpt('./save_model/10epoch/mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth')
    if cnn_mix:
        models_to_test.append((cnn_mix, 'Mix-AT'))

    if not models_to_test:
        print("[ERROR] 没有加载到任何模型，请检查模型路径")
        exit(1)

    # 3. 运行所有测试
    print("\n[3/4] 运行所有遮蔽攻击测试...")
    print("-"*60)
    all_results = []
    for model, name in models_to_test:
        print(f"\n测试模型: {name}")
        results = run_all_occlusion_tests(model, name, imgs, lbls)
        all_results.append(results)

    # 4. 绘制折线图
    print("\n[4/4] 绘制折线图...")
    print("-"*60)
    plot_occlusion_line_charts(all_results)

    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)
