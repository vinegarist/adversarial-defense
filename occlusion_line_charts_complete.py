"""
========== 遮蔽攻击折线图可视化（完整版 - 加载所有模型） ==========
内核重启后可直接运行，加载所有训练过的模型，测试所有参数配置
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


def load_model_from_ckpt(ckpt_path, model_name="Model"):
    """从检查点加载模型"""
    if not os.path.exists(ckpt_path):
        print(f"[WARN] 模型不存在: {ckpt_path}")
        return None
    net = LeNet5().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state['net'])
    net.eval()
    print(f"[OK] {model_name}: {os.path.basename(ckpt_path)}")
    return net


def get_all_models():
    """获取所有要测试的模型配置"""
    models_config = []

    # 1. 标准模型
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5.pth',
        'name': 'Standard'
    })

    # 2. PGD对抗训练
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5_PGD_0.1_5_AT.pth',
        'name': 'PGD-AT'
    })

    # 3. FGSM对抗训练
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5_FGSM_AT.pth',
        'name': 'FGSM-AT'
    })

    # 4. Occlusion对抗训练 (Fixed)
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5_OcclusionAT_9_3.pth',
        'name': 'Occlusion-AT'
    })

    # 5. Adaptive Occlusion对抗训练
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5_AdaptiveOcclusionAT_5_3.pth',
        'name': 'Adaptive-Occlusion-AT'
    })

    # 6. Adaptive Saliency对抗训练 (N=5,R=1)
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_1.pth',
        'name': 'Adaptive-Saliency-AT(N5R1)'
    })

    # 7. Adaptive Saliency对抗训练 (N=5,R=3) - 主要模型
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth',
        'name': 'Adaptive-Saliency-AT(N5R3)'
    })

    # 8. Adaptive Mixed对抗训练
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5_AdaptiveMixedAT_0.5_5_3.pth',
        'name': 'Adaptive-Mixed-AT'
    })

    # 9. Mixed Occlusion+PGD (10epoch)
    models_config.append({
        'path': './save_model/10epoch/mnist_lenet5_MixedOcclusionPgdAT_0.5_9_3.pth',
        'name': 'Mixed-Occlusion-PGD-AT(10e)'
    })

    # 10. Adaptive Saliency + PGD Mixed (50epoch)
    models_config.append({
        'path': './save_model/50epoch/mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth',
        'name': 'Adaptive-Saliency-PGD-MixAT'
    })

    # 11-15. Free-AT (不同minibatch数量)
    for m in [2, 5, 10, 25, 50]:
        path = f'./save_model/50epoch/mnist_lenet5_Free_AT_{m}.pth'
        if os.path.exists(path):
            models_config.append({
                'path': path,
                'name': f'Free-AT(m={m})'
            })

    return models_config


def run_all_occlusion_tests(model, model_name, imgs, lbls):
    """
    运行所有遮蔽攻击测试，返回结果字典

    完整参数配置:
    - Fixed: top_k = [3, 5, 7, 9, 12, 15], kernel_size=3 (radius=1)
    - Adaptive N: N = [3, 5, 7, 10], R=3
    - Adaptive R: R = [2, 3, 4], N=5
    """
    results = {'model_name': model_name}

    # 1. 干净样本准确率
    print(f"\n  [{model_name}] 测试干净样本...")
    clean_acc, _ = test_fn(model, imgs, lbls, bs=250, mode='clean')
    results['Clean'] = clean_acc
    print(f"    Clean: {clean_acc:.2f}%")

    # 2. Fixed遮蔽攻击 - 不同top_k (固定kernel_size=3, 即radius=1)
    print(f"  [{model_name}] Fixed-Saliency测试 (k变化)...")
    top_k_values = [3, 5, 7, 9, 12, 15]
    for top_k in top_k_values:
        attack = SaliencyOcclusionAttack(model, top_k=top_k, kernel_size=3, occlu_color=0.0)
        acc, _ = test_fn(nn.Sequential(attack, model), imgs, lbls, bs=250, mode='attack')
        results[f'Fixed_k{top_k}'] = acc
        print(f"    k={top_k:2d}: {acc:6.2f}%")

    # 3. Fixed遮蔽攻击 - 不同kernel_size (固定top_k=9)
    # kernel_size=3 -> radius=1, kernel_size=5 -> radius=2, kernel_size=7 -> radius=3
    print(f"  [{model_name}] Fixed-Saliency测试 (radius变化)...")
    kernel_sizes = [3, 5, 7]  # radius = 1, 2, 3
    top_k_fixed = 9
    for ks in kernel_sizes:
        attack = SaliencyOcclusionAttack(model, top_k=top_k_fixed, kernel_size=ks, occlu_color=0.0)
        acc, _ = test_fn(nn.Sequential(attack, model), imgs, lbls, bs=250, mode='attack')
        radius = ks // 2
        results[f'Fixed_r{radius}'] = acc
        print(f"    r={radius} (ks={ks}): {acc:.2f}%")

    # 4. Adaptive遮蔽攻击 - 不同N (固定R=3)
    print(f"  [{model_name}] Adaptive-Saliency测试 (N变化)...")
    N_values = [3, 5, 7, 10]
    R_fixed = 3
    for N in N_values:
        attack = AdaptiveSaliencyOcclusionAttack(model, N=N, R=R_fixed, c=0.0)
        acc, _ = test_fn(nn.Sequential(attack, model), imgs, lbls, bs=250, mode='attack')
        results[f'Adaptive_N{N}_R{R_fixed}'] = acc
        print(f"    N={N:2d},R={R_fixed}: {acc:6.2f}%")

    # 5. Adaptive遮蔽攻击 - 不同R (固定N=5)
    print(f"  [{model_name}] Adaptive-Saliency测试 (R变化)...")
    N_fixed = 5
    R_values = [2, 3, 4]
    for R in R_values:
        attack = AdaptiveSaliencyOcclusionAttack(model, N=N_fixed, R=R, c=0.0)
        acc, _ = test_fn(nn.Sequential(attack, model), imgs, lbls, bs=250, mode='attack')
        results[f'Adaptive_N{N_fixed}_R{R}'] = acc
        print(f"    N={N_fixed},R={R}: {acc:6.2f}%")

    return results


def plot_all_line_charts(all_results, save_dir='./results_figures'):
    """
    绘制所有折线图
    """
    os.makedirs(save_dir, exist_ok=True)

    models = [r['model_name'] for r in all_results]
    n_models = len(models)

    # 颜色配置 - 支持更多模型
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                   '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']
    colors = (base_colors * ((n_models // len(base_colors)) + 1))[:n_models]

    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h',
               'x', '+', '1', '2', '3', '4']

    # ========== 图1: Fixed-Saliency - k值影响 (所有k值) ==========
    fig, ax = plt.subplots(figsize=(14, 7))

    k_values = [3, 5, 7, 9, 12, 15]

    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Fixed_k{k}', 0) for k in k_values]
        ax.plot(k_values, acc_values,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2.5, markersize=10,
                label=result['model_name'])

    # 添加Clean基线（虚线）
    for idx, result in enumerate(all_results):
        ax.axhline(y=result['Clean'], color=colors[idx % len(colors)],
                   linestyle='--', alpha=0.2, linewidth=1)

    ax.set_xlabel('k (遮蔽区域数量)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Fixed-Saliency攻击: k值影响 (kernel_size=3)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', ncol=2 if n_models > 8 else 1)
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
                'Parameter': 'k',
                'k': k,
                'radius': 1,
                'kernel_size': 3,
                'Accuracy': result.get(f'Fixed_k{k}', 0)
            })
    pd.DataFrame(csv_data).to_csv(f'{save_dir}/data_fixed_saliency_k_complete.csv', index=False)

    # ========== 图2: Fixed-Saliency - radius影响 ==========
    fig, ax = plt.subplots(figsize=(12, 6))

    radius_values = [1, 2, 3]

    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Fixed_r{r}', 0) for r in radius_values]
        ax.plot(radius_values, acc_values,
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2.5, markersize=10,
                label=result['model_name'])

    ax.set_xlabel('Radius (遮蔽半径)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title('Fixed-Saliency攻击: Radius影响 (k=9)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 105)
    ax.set_xticks(radius_values)

    plt.tight_layout()
    save_path = f'{save_dir}/line_fixed_saliency_radius_complete.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[SAVED] {save_path}')

    # 保存CSV
    csv_data = []
    for result in all_results:
        for r in radius_values:
            csv_data.append({
                'Model': result['model_name'],
                'Attack_Type': 'Fixed-Saliency',
                'Parameter': 'radius',
                'k': 9,
                'radius': r,
                'kernel_size': r * 2 + 1,
                'Accuracy': result.get(f'Fixed_r{r}', 0)
            })
    pd.DataFrame(csv_data).to_csv(f'{save_dir}/data_fixed_saliency_radius_complete.csv', index=False)

    # ========== 图3: Adaptive-Saliency - N值影响 (R=3) ==========
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
    ax.legend(fontsize=10, loc='upper right')
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
                'Parameter': 'N',
                'N': N,
                'R': R_fixed,
                'Accuracy': result.get(f'Adaptive_N{N}_R{R_fixed}', 0)
            })
    pd.DataFrame(csv_data).to_csv(f'{save_dir}/data_adaptive_saliency_N_complete.csv', index=False)

    # ========== 图4: Adaptive-Saliency - R值影响 (N=5) ==========
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
    ax.legend(fontsize=10, loc='upper right')
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
                'Parameter': 'R',
                'N': N_fixed,
                'R': R,
                'Accuracy': result.get(f'Adaptive_N{N_fixed}_R{R}', 0)
            })
    pd.DataFrame(csv_data).to_csv(f'{save_dir}/data_adaptive_saliency_R_complete.csv', index=False)

    # ========== 图5: 综合对比 - 所有攻击类型 (2x2子图) ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: Fixed k
    k_values = [3, 5, 7, 9, 12, 15]
    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Fixed_k{k}', 0) for k in k_values]
        axes[0, 0].plot(k_values, acc_values,
                        marker=markers[idx % len(markers)],
                        color=colors[idx % len(colors)],
                        linewidth=2, markersize=8,
                        label=result['model_name'])
    axes[0, 0].set_xlabel('k (遮蔽区域数量)', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Fixed-Saliency: k值影响', fontsize=14)
    axes[0, 0].legend(fontsize=8, loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 105)

    # 子图2: Fixed radius
    radius_values = [1, 2, 3]
    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Fixed_r{r}', 0) for r in radius_values]
        axes[0, 1].plot(radius_values, acc_values,
                        marker=markers[idx % len(markers)],
                        color=colors[idx % len(colors)],
                        linewidth=2, markersize=8,
                        label=result['model_name'])
    axes[0, 1].set_xlabel('Radius (遮蔽半径)', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].set_title('Fixed-Saliency: Radius影响', fontsize=14)
    axes[0, 1].legend(fontsize=8, loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 105)

    # 子图3: Adaptive N
    N_values = [3, 5, 7, 10]
    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Adaptive_N{N}_R3', 0) for N in N_values]
        axes[1, 0].plot(N_values, acc_values,
                        marker=markers[idx % len(markers)],
                        color=colors[idx % len(colors)],
                        linewidth=2, markersize=8,
                        label=result['model_name'])
    axes[1, 0].set_xlabel('N (最大遮蔽区域数)', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Adaptive-Saliency: N值影响 (R=3)', fontsize=14)
    axes[1, 0].legend(fontsize=8, loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 105)

    # 子图4: Adaptive R
    R_values = [2, 3, 4]
    for idx, result in enumerate(all_results):
        acc_values = [result.get(f'Adaptive_N5_R{R}', 0) for R in R_values]
        axes[1, 1].plot(R_values, acc_values,
                        marker=markers[idx % len(markers)],
                        color=colors[idx % len(colors)],
                        linewidth=2, markersize=8,
                        label=result['model_name'])
    axes[1, 1].set_xlabel('R (遮蔽半径)', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Adaptive-Saliency: R值影响 (N=5)', fontsize=14)
    axes[1, 1].legend(fontsize=8, loc='upper right')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 105)

    plt.tight_layout()
    save_path = f'{save_dir}/line_all_occlusion_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[SAVED] {save_path}')

    # 保存汇总CSV
    csv_data = []
    for result in all_results:
        # Clean
        csv_data.append({
            'Model': result['model_name'],
            'Attack_Type': 'Clean',
            'Param': '-',
            'Accuracy': result['Clean']
        })
        # Fixed k
        for k in k_values:
            csv_data.append({
                'Model': result['model_name'],
                'Attack_Type': 'Fixed-Saliency',
                'Param': f'k={k}',
                'Accuracy': result.get(f'Fixed_k{k}', 0)
            })
        # Fixed radius
        for r in radius_values:
            csv_data.append({
                'Model': result['model_name'],
                'Attack_Type': 'Fixed-Saliency',
                'Param': f'r={r}',
                'Accuracy': result.get(f'Fixed_r{r}', 0)
            })
        # Adaptive N
        for N in N_values:
            csv_data.append({
                'Model': result['model_name'],
                'Attack_Type': 'Adaptive-Saliency',
                'Param': f'N={N},R=3',
                'Accuracy': result.get(f'Adaptive_N{N}_R3', 0)
            })
        # Adaptive R
        for R in R_values:
            csv_data.append({
                'Model': result['model_name'],
                'Attack_Type': 'Adaptive-Saliency',
                'Param': f'N=5,R={R}',
                'Accuracy': result.get(f'Adaptive_N5_R{R}', 0)
            })

    pd.DataFrame(csv_data).to_csv(f'{save_dir}/all_occlusion_results_summary.csv', index=False)
    print(f'[SAVED] {save_dir}/all_occlusion_results_summary.csv')

    print('\n' + '='*60)
    print('所有折线图绘制完成！')
    print('='*60)


# ========== 主执行 ==========
if __name__ == '__main__':
    print("\n" + "="*70)
    print("遮蔽攻击折线图可视化 - 完整版（加载所有模型）")
    print("="*70)

    # 1. 加载测试数据
    print("\n[1/4] 加载测试数据...")
    imgs, lbls = load_mnist_test()
    print(f"      测试集大小: {len(imgs)}")

    # 2. 获取并加载所有模型
    print("\n[2/4] 加载所有模型...")
    models_config = get_all_models()
    print(f"      找到 {len(models_config)} 个模型配置")
    print("-" * 50)

    loaded_models = []
    for cfg in models_config:
        model = load_model_from_ckpt(cfg['path'], cfg['name'])
        if model:
            loaded_models.append((model, cfg['name']))

    if not loaded_models:
        print("[ERROR] 没有加载到任何模型！")
        exit(1)

    print(f"\n[OK] 成功加载 {len(loaded_models)} 个模型")
    print("-" * 50)

    # 3. 运行所有测试
    print("\n[3/4] 运行所有遮蔽攻击测试...")
    print("=" * 50)
    all_results = []
    for model, name in loaded_models:
        results = run_all_occlusion_tests(model, name, imgs, lbls)
        all_results.append(results)

    # 4. 绘制折线图
    print("\n[4/4] 绘制折线图...")
    print("=" * 50)
    plot_all_line_charts(all_results)

    print("\n" + "="*70)
    print("全部完成！")
    print("="*70)
