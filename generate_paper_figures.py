#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文图表生成脚本
用途: 生成第四章实验所需的全部可视化图表
环境: conda activate adv-attack
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets, transforms
import copy

# 导入本地模块
from models import LeNet5
from occlusion_attack import (
    compute_saliency,
    SaliencyOcclusionAttack,
    AdaptiveSaliencyOcclusionAttack
)

# ============================================================
# 配置参数
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'paper_figures')
THESIS_FIGURES_DIR = r'D:\软件\南开大学论文模板2026\figures'
MODEL_DIR = os.path.join(BASE_DIR, 'save_model', '50epoch')
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_figures')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 关键参数 (与论文一致)
N = 5  # 最大遮蔽区域数
R = 3  # 最大遮蔽半径

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# 专业配色方案
COLORS = {
    'standard': '#FF6B6B',      # 红色 - 标准模型
    'at': '#4ECDC4',            # 青色 - AT模型
    'mix_at': '#45B7D1',        # 蓝色 - 混合AT模型
    'clean': '#2ECC71',         # 绿色 - 干净样本
    'attack': '#E74C3C',        # 红色 - 攻击
    'train_acc': '#3498DB',     # 蓝色 - 训练准确率
    'test_acc': '#E67E22',      # 橙色 - 测试准确率
    'train_loss': '#9B59B6',    # 紫色 - 训练损失
}

# ============================================================
# 工具函数
# ============================================================
def load_model(model_path):
    """加载模型"""
    model = LeNet5()
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    # 支持两种格式：直接state_dict 或 包含'net'键的字典
    if isinstance(checkpoint, dict) and 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

def get_test_loader(batch_size=100, shuffle=True):
    """获取测试数据加载器"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=shuffle)
    return loader

def get_test_samples(n_samples=10, seed=42):
    """获取测试样本（固定种子确保可复现）"""
    torch.manual_seed(seed)
    loader = get_test_loader(batch_size=n_samples, shuffle=True)
    images, labels = next(iter(loader))
    return images.to(DEVICE), labels.to(DEVICE)

def imshow_with_pred(ax, img, title='', cmap='gray'):
    """显示图像并设置标题"""
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.transpose(1, 2, 0).squeeze()
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

# ============================================================
# 图表1: 自适应遮蔽攻击可视化
# ============================================================
def generate_fig4_1_adaptive_occlusion_viz():
    """生成图4.1: 自适应遮蔽攻击可视化"""
    print("生成图4.1: 自适应遮蔽攻击可视化...")

    # 加载标准模型
    std_model = load_model(os.path.join(MODEL_DIR, 'mnist_lenet5.pth'))

    # 获取测试样本
    images, labels = get_test_samples(n_samples=6, seed=123)

    # 创建攻击实例
    attack_n1 = AdaptiveSaliencyOcclusionAttack(std_model, N=1, R=3, c=0.0)
    attack_n3 = AdaptiveSaliencyOcclusionAttack(std_model, N=3, R=3, c=0.0)
    attack_n5 = AdaptiveSaliencyOcclusionAttack(std_model, N=5, R=3, c=0.0)

    # 创建图表
    fig, axes = plt.subplots(6, 5, figsize=(15, 12))

    for i in range(6):
        img = images[i:i+1]
        label = labels[i:i+1]

        # 计算显著性图
        img_grad = img.detach().requires_grad_(True)
        saliency = compute_saliency(std_model, img_grad, label)
        saliency_map = saliency[0, 0].cpu().detach().numpy()

        # 执行不同N值的攻击
        adv_n1 = attack_n1((img, label))
        adv_n3 = attack_n3((img, label))
        adv_n5 = attack_n5((img, label))

        # 显示
        imshow_with_pred(axes[i, 0], img[0], f'原图 (标签:{label.item()})')
        imshow_with_pred(axes[i, 1], saliency_map, '显著性图', cmap='hot')
        imshow_with_pred(axes[i, 2], adv_n1[0], f'N=1遮蔽')
        imshow_with_pred(axes[i, 3], adv_n3[0], f'N=3遮蔽')
        imshow_with_pred(axes[i, 4], adv_n5[0], f'N=5遮蔽')

    plt.suptitle('图4.1 自适应显著性遮蔽攻击效果可视化\n(从左至右: 原始图像 → 显著性图 → 不同N值遮蔽效果)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 保存
    save_path = os.path.join(FIGURES_DIR, 'fig4_1_adaptive_occlusion_viz.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  已保存: {save_path}")
    return save_path

# ============================================================
# 图表2: 训练曲线对比
# ============================================================
def generate_fig4_2_training_curves():
    """生成图4.2: 训练曲线对比"""
    print("生成图4.2: 训练曲线对比...")

    # 读取训练历史数据
    saliency_history_path = os.path.join(RESULTS_DIR, 'adaptive_saliency_at_training_history_5_1.csv')
    ig_history_path = os.path.join(RESULTS_DIR, 'adaptive_ig_at_training_history_5_3.csv')

    saliency_df = pd.read_csv(saliency_history_path)
    ig_df = pd.read_csv(ig_history_path)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: Saliency-AT (N=5, R=1)
    ax1 = axes[0]
    epochs = saliency_df['epoch'].values
    ax1.plot(epochs, saliency_df['train_acc'].values, 'b-', label='训练准确率', linewidth=2)
    ax1.plot(epochs, saliency_df['test_acc'].values, 'g--', label='测试准确率(鲁棒)', linewidth=2)
    ax1.plot(epochs, saliency_df['test_clean_acc'].values, 'r-.', label='干净样本准确率', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('准确率 (%)', fontsize=11)
    ax1.set_title('(a) Adaptive-Saliency-AT (N=5, R=1)', fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([95, 100])

    # 子图2: IG-AT (N=5, R=3)
    ax2 = axes[1]
    epochs = ig_df['epoch'].values
    ax2.plot(epochs, ig_df['train_acc'].values, 'b-', label='训练准确率', linewidth=2)
    ax2.plot(epochs, ig_df['test_acc'].values, 'g--', label='测试准确率(鲁棒)', linewidth=2)
    ax2.plot(epochs, ig_df['test_clean_acc'].values, 'r-.', label='干净样本准确率', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('准确率 (%)', fontsize=11)
    ax2.set_title('(b) Adaptive-IG-AT (N=5, R=3)', fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('图4.2 对抗性训练过程曲线', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 保存
    save_path = os.path.join(FIGURES_DIR, 'fig4_2_training_curves.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  已保存: {save_path}")
    return save_path

# ============================================================
# 图表3: 攻击强度对比柱状图
# ============================================================
def generate_fig4_3_attack_comparison():
    """生成图4.3: 攻击强度对比柱状图"""
    print("生成图4.3: 攻击强度对比...")

    # 读取评估数据
    eval_path = os.path.join(RESULTS_DIR, 'data_adaptive_saliency_N_complete.csv')
    if os.path.exists(eval_path):
        df_raw = pd.read_csv(eval_path)
        # 转换数据格式
        n_values = sorted(df_raw['N'].unique())
        standard_accs = df_raw[df_raw['Model'] == 'Standard']['Accuracy'].values
        at_accs = df_raw[df_raw['Model'] == 'Adaptive-Saliency-AT']['Accuracy'].values
    else:
        # 使用备用数据
        n_values = [3, 5, 7, 10]
        standard_accs = [49.45, 34.33, 25.09, 15.95]
        at_accs = [95.97, 94.3, 92.44, 88.74]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(n_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, standard_accs, width, label='标准模型', color=COLORS['standard'])
    bars2 = ax.bar(x + width/2, at_accs, width, label='Adaptive-Saliency-AT', color=COLORS['at'])

    ax.set_xlabel('遮蔽区域数 N', fontsize=11)
    ax.set_ylabel('准确率 (%)', fontsize=11)
    ax.set_title('图4.3 不同攻击强度下的模型准确率对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in n_values])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # 保存
    save_path = os.path.join(FIGURES_DIR, 'fig4_3_attack_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  已保存: {save_path}")
    return save_path

# ============================================================
# 图表4: 防御策略热力图
# ============================================================
def generate_fig4_4_defense_heatmap():
    """生成图4.4: 防御策略热力图"""
    print("生成图4.4: 防御策略热力图...")

    # 读取多模型对比数据
    eval_path = os.path.join(RESULTS_DIR, 'all_models_all_attacks_evaluation.csv')
    df = pd.read_csv(eval_path)

    # 准备数据
    models = df['Model'].values
    attacks = ['Clean', 'FGSM', 'PGD(Linf)', 'CW(Linf)', 'Occl(Fixed)', 'Occl(Adaptive)']

    # 创建数据矩阵
    data = df[attacks].values

    # 创建热力图
    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # 设置刻度
    ax.set_xticks(np.arange(len(attacks)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(attacks, fontsize=10)
    ax.set_yticklabels(models, fontsize=9)

    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 添加数值标注
    for i in range(len(models)):
        for j in range(len(attacks)):
            text = ax.text(j, i, f'{data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title('图4.4 不同防御策略对多种攻击的鲁棒性对比 (%)', fontsize=12, fontweight='bold')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('准确率 (%)', rotation=-90, va="bottom", fontsize=10)

    plt.tight_layout()

    # 保存
    save_path = os.path.join(FIGURES_DIR, 'fig4_4_defense_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  已保存: {save_path}")
    return save_path

# ============================================================
# 图表5: 参数敏感性分析
# ============================================================
def generate_fig4_5_param_sensitivity():
    """生成图4.5: 参数敏感性分析"""
    print("生成图4.5: 参数敏感性分析...")

    # 读取参数敏感性数据
    n_path = os.path.join(RESULTS_DIR, 'data_adaptive_saliency_N_complete.csv')
    r_path = os.path.join(RESULTS_DIR, 'data_adaptive_saliency_R_complete.csv')
    k_path = os.path.join(RESULTS_DIR, 'data_fixed_saliency_k_complete.csv')
    color_path = os.path.join(RESULTS_DIR, 'eval_color_param_N5R3.csv')

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 子图1: N参数影响
    ax1 = axes[0, 0]
    if os.path.exists(n_path):
        df_n = pd.read_csv(n_path)
        n_vals = sorted(df_n['N'].unique())
        std_accs = df_n[df_n['Model'] == 'Standard']['Accuracy'].values
        at_accs = df_n[df_n['Model'] == 'Adaptive-Saliency-AT']['Accuracy'].values
        ax1.plot(n_vals, std_accs, 'ro-', linewidth=2, markersize=8, label='标准模型')
        ax1.plot(n_vals, at_accs, 'go-', linewidth=2, markersize=8, label='AT模型')
        ax1.legend(loc='lower left', fontsize=9)
    else:
        n_vals = [3, 5, 7, 10]
        accs = [49.45, 34.33, 25.09, 15.95]
        ax1.plot(n_vals, accs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('遮蔽区域数 N', fontsize=11)
    ax1.set_ylabel('准确率 (%)', fontsize=11)
    ax1.set_title('(a) N参数影响 (R=3固定)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 子图2: R参数影响
    ax2 = axes[0, 1]
    if os.path.exists(r_path):
        df_r = pd.read_csv(r_path)
        r_vals = sorted(df_r['R'].unique())
        std_accs = df_r[df_r['Model'] == 'Standard']['Accuracy'].values
        at_accs = df_r[df_r['Model'] == 'Adaptive-Saliency-AT']['Accuracy'].values
        ax2.plot(r_vals, std_accs, 'ro-', linewidth=2, markersize=8, label='标准模型')
        ax2.plot(r_vals, at_accs, 'go-', linewidth=2, markersize=8, label='AT模型')
        ax2.legend(loc='lower left', fontsize=9)
    else:
        r_vals = [2, 3, 4]
        accs = [49.15, 34.33, 23.91]
        ax2.plot(r_vals, accs, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('遮蔽半径 R', fontsize=11)
    ax2.set_ylabel('准确率 (%)', fontsize=11)
    ax2.set_title('(b) R参数影响 (N=5固定)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 子图3: top_k参数影响
    ax3 = axes[1, 0]
    if os.path.exists(k_path):
        df_k = pd.read_csv(k_path)
        k_vals = sorted(df_k['k'].unique())
        std_accs = df_k[df_k['Model'] == 'Standard']['Accuracy'].values
        at_accs = df_k[df_k['Model'] == 'Adaptive-Saliency-AT']['Accuracy'].values
        ax3.plot(k_vals, std_accs, 'ro-', linewidth=2, markersize=8, label='标准模型')
        ax3.plot(k_vals, at_accs, 'go-', linewidth=2, markersize=8, label='AT模型')
        ax3.legend(loc='lower left', fontsize=9)
    else:
        k_vals = [3, 5, 9, 15]
        accs = [97.79, 96.97, 95.48, 92.56]
        ax3.plot(k_vals, accs, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('固定遮蔽区域数 top_k', fontsize=11)
    ax3.set_ylabel('准确率 (%)', fontsize=11)
    ax3.set_title('(c) 固定遮蔽top_k参数影响', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 子图4: 遮蔽颜色影响
    ax4 = axes[1, 1]
    if os.path.exists(color_path):
        df_color = pd.read_csv(color_path)
        colors_name = df_color['color'].values if 'color' in df_color.columns else ['黑色', '灰色', '白色']
        accs_color = df_color['accuracy'].values if 'accuracy' in df_color.columns else [80.65, 77.88, 54.77]
    else:
        colors_name = ['黑色(0.0)', '灰色(0.5)', '白色(1.0)']
        accs_color = [80.65, 77.88, 54.77]
    bars = ax4.bar(range(len(colors_name)), accs_color, color=['black', 'gray', 'white'][:len(colors_name)], edgecolor='black', linewidth=1.5)
    ax4.set_xticks(range(len(colors_name)))
    ax4.set_xticklabels(colors_name)
    ax4.set_xlabel('遮蔽颜色', fontsize=11)
    ax4.set_ylabel('准确率 (%)', fontsize=11)
    ax4.set_title('(d) 遮蔽颜色影响 (N=5, R=3)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    plt.suptitle('图4.5 参数敏感性分析', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 保存
    save_path = os.path.join(FIGURES_DIR, 'fig4_5_param_sensitivity.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  已保存: {save_path}")
    return save_path

# ============================================================
# 图表6: Mask对比可视化
# ============================================================
def generate_fig4_6_mask_comparison():
    """生成图4.6: 不同参数Mask对比"""
    print("生成图4.6: Mask对比可视化...")

    # 加载标准模型
    std_model = load_model(os.path.join(MODEL_DIR, 'mnist_lenet5.pth'))

    # 获取测试样本
    images, labels = get_test_samples(n_samples=5, seed=456)

    # 不同参数组合
    param_combos = [(1, 1), (3, 2), (5, 3), (7, 3)]

    # 创建图表
    fig, axes = plt.subplots(5, 5, figsize=(14, 12))

    for i in range(5):
        img = images[i:i+1]
        label = labels[i:i+1]

        # 显示原图
        imshow_with_pred(axes[i, 0], img[0], f'原图 (标签:{label.item()})')

        # 显示不同参数的遮蔽效果
        for j, (n, r) in enumerate(param_combos):
            attack = AdaptiveSaliencyOcclusionAttack(std_model, N=n, R=r, c=0.0)
            adv = attack((img, label))
            imshow_with_pred(axes[i, j+1], adv[0], f'N={n}, R={r}')

    plt.suptitle('图4.6 不同(N,R)参数组合下的遮蔽效果对比', fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 保存
    save_path = os.path.join(FIGURES_DIR, 'fig4_6_mask_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  已保存: {save_path}")
    return save_path

# ============================================================
# 图表7: 标准模型vs AT模型显著性对比
# ============================================================
def generate_fig4_7_saliency_comparison():
    """生成图4.7: 标准模型vs AT模型显著性对比"""
    print("生成图4.7: 显著性对比...")

    # 加载模型
    std_model = load_model(os.path.join(MODEL_DIR, 'mnist_lenet5.pth'))
    at_model = load_model(os.path.join(MODEL_DIR, 'mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth'))

    # 获取测试样本
    images, labels = get_test_samples(n_samples=6, seed=789)

    # 创建图表
    fig, axes = plt.subplots(6, 3, figsize=(10, 14))

    for i in range(6):
        img = images[i:i+1]
        label = labels[i:i+1]

        # 计算标准模型的显著性
        img_grad_std = img.detach().requires_grad_(True)
        saliency_std = compute_saliency(std_model, img_grad_std, label)
        saliency_std_map = saliency_std[0, 0].cpu().detach().numpy()

        # 计算AT模型的显著性
        img_grad_at = img.detach().requires_grad_(True)
        saliency_at = compute_saliency(at_model, img_grad_at, label)
        saliency_at_map = saliency_at[0, 0].cpu().detach().numpy()

        # 显示
        imshow_with_pred(axes[i, 0], img[0], f'原图 (标签:{label.item()})')
        imshow_with_pred(axes[i, 1], saliency_std_map, '标准模型显著性', cmap='hot')
        imshow_with_pred(axes[i, 2], saliency_at_map, 'AT模型显著性', cmap='hot')

    plt.suptitle('图4.7 标准模型与对抗性训练模型的显著性图对比\n(展示对抗性训练如何改变模型的关注区域)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    # 保存
    save_path = os.path.join(FIGURES_DIR, 'fig4_7_saliency_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  已保存: {save_path}")
    return save_path

# ============================================================
# 图表8: Mix-AT综合性能雷达图
# ============================================================
def generate_fig4_8_radar_chart():
    """生成图4.8: Mix-AT综合性能雷达图"""
    print("生成图4.8: Mix-AT综合性能雷达图...")

    # 读取评估数据
    eval_path = os.path.join(RESULTS_DIR, 'all_models_all_attacks_evaluation.csv')
    df = pd.read_csv(eval_path)

    # 选择要展示的维度
    categories = ['Clean', 'FGSM', 'PGD(Linf)', 'CW(Linf)', 'Occl(Fixed)', 'Occl(Adaptive)']

    # 选择要对比的模型
    models_to_compare = ['Standard', 'PGD-AT', 'Occlusion-AT(Adaptive)', 'Mix-AT(Adaptive,50ep)']

    # 创建雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors_radar = ['#FF6B6B', '#3498DB', '#2ECC71', '#F39C12']

    for idx, model in enumerate(models_to_compare):
        if model in df['Model'].values:
            values = df[df['Model'] == model][categories].values[0].tolist()
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[idx])
            ax.fill(angles, values, alpha=0.25, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.title('图4.8 不同防御策略的综合性能雷达图', fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()

    # 保存
    save_path = os.path.join(FIGURES_DIR, 'fig4_8_radar_chart.png')
    plt.savefig(save_path)
    plt.close()
    print(f"  已保存: {save_path}")
    return save_path

# ============================================================
# 主函数
# ============================================================
def main():
    print("="*60)
    print("开始生成论文图表...")
    print(f"输出目录: {FIGURES_DIR}")
    print(f"设备: {DEVICE}")
    print("="*60)

    # 创建输出目录
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(THESIS_FIGURES_DIR, exist_ok=True)

    # 生成所有图表
    saved_files = []

    try:
        saved_files.append(generate_fig4_1_adaptive_occlusion_viz())
    except Exception as e:
        print(f"  图4.1生成失败: {e}")

    try:
        saved_files.append(generate_fig4_2_training_curves())
    except Exception as e:
        print(f"  图4.2生成失败: {e}")

    try:
        saved_files.append(generate_fig4_3_attack_comparison())
    except Exception as e:
        print(f"  图4.3生成失败: {e}")

    try:
        saved_files.append(generate_fig4_4_defense_heatmap())
    except Exception as e:
        print(f"  图4.4生成失败: {e}")

    try:
        saved_files.append(generate_fig4_5_param_sensitivity())
    except Exception as e:
        print(f"  图4.5生成失败: {e}")

    try:
        saved_files.append(generate_fig4_6_mask_comparison())
    except Exception as e:
        print(f"  图4.6生成失败: {e}")

    try:
        saved_files.append(generate_fig4_7_saliency_comparison())
    except Exception as e:
        print(f"  图4.7生成失败: {e}")

    try:
        saved_files.append(generate_fig4_8_radar_chart())
    except Exception as e:
        print(f"  图4.8生成失败: {e}")

    # 复制到论文目录
    print("\n复制图表到论文模板目录...")
    import shutil
    for f in saved_files:
        if f and os.path.exists(f):
            dest = os.path.join(THESIS_FIGURES_DIR, os.path.basename(f))
            shutil.copy(f, dest)
            print(f"  已复制: {os.path.basename(f)}")

    print("\n" + "="*60)
    print(f"完成! 共生成 {len([f for f in saved_files if f])} 张图表")
    print(f"图表保存在: {FIGURES_DIR}")
    print(f"已复制到: {THESIS_FIGURES_DIR}")
    print("="*60)

if __name__ == '__main__':
    main()
