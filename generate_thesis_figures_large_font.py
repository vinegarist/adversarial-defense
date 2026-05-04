# -*- coding: utf-8 -*-
"""
论文图表生成脚本 - 大字体版本
将 Jupyter notebook 实验成果整理成论文图表，字体增大以便在论文中更清晰可读

使用方法:
    conda activate adv-attack
    python generate_thesis_figures_large_font.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 配置路径
DATA_DIR = Path(r"d:\软件\对抗性防御\对抗性防御-1\03.代码\results_figures")
MODEL_DIR = Path(r"d:\软件\对抗性防御\对抗性防御-1\03.代码\save_model\50epoch")
THESIS_DIR = Path(r"D:\软件\南开大学论文模板2026")
OUTPUT_DIR = THESIS_DIR / "figures"

# 确保输出目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 大字体配置 - 主要修改部分
# =============================================================================
# 基础字体大小从12增大到16
BASE_FONT_SIZE = 20
# 标题字体大小从14增大到20
TITLE_FONT_SIZE = 26
# 轴标签字体大小从14增大到18
LABEL_FONT_SIZE = 24
# 刻度字体大小
TICK_FONT_SIZE = 20
# 图例字体大小从10增大到14
LEGEND_FONT_SIZE = 20
# 注释字体大小
ANNOT_FONT_SIZE = 18

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = BASE_FONT_SIZE
matplotlib.rcParams['axes.labelsize'] = LABEL_FONT_SIZE
matplotlib.rcParams['axes.titlesize'] = TITLE_FONT_SIZE
matplotlib.rcParams['legend.fontsize'] = LEGEND_FONT_SIZE
matplotlib.rcParams['xtick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = TICK_FONT_SIZE
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

# 颜色配置
COLORS = {
    'standard': '#7F7F7F',
    'pgd_at': '#1f77b4',
    'adaptive_saliency': '#2ca02c',
    'mix_at': '#ff7f0e',
    'highlight': '#d62728',
    'fixed': '#3498db',
    'adaptive': '#e74c3c',
    'ig': '#9b59b6',
}


class LeNet5(nn.Module):
    """LeNet-5 网络结构"""
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ThesisFigureGenerator:
    def __init__(self):
        self.data = {}
        self.load_all_data()

    def load_all_data(self):
        """加载所有数据文件"""
        print("正在加载数据文件...")

        # 训练历史
        training_history_path = DATA_DIR / "adaptive_ig_at_training_history_5_3.csv"
        if training_history_path.exists():
            self.data['training_history'] = pd.read_csv(training_history_path)
            print(f"  - 训练历史: {len(self.data['training_history'])} 条记录")

        # 综合对比数据
        comprehensive_path = DATA_DIR / "comprehensive_multi_model_comparison.csv"
        if comprehensive_path.exists():
            self.data['comprehensive'] = pd.read_csv(comprehensive_path)
            print(f"  - 综合对比: {len(self.data['comprehensive'])} 个模型")

        # N参数敏感性
        n_param_path = DATA_DIR / "data_adaptive_saliency_N_complete.csv"
        if n_param_path.exists():
            self.data['n_param'] = pd.read_csv(n_param_path)
            print(f"  - N参数数据: {len(self.data['n_param'])} 条记录")

        # R参数敏感性
        r_param_path = DATA_DIR / "data_adaptive_saliency_R_complete.csv"
        if r_param_path.exists():
            self.data['r_param'] = pd.read_csv(r_param_path)
            print(f"  - R参数数据: {len(self.data['r_param'])} 条记录")

        # 白盒与迁移攻击对比
        transfer_path = DATA_DIR / "whitebox_vs_transfer_attack_comparison.csv"
        if transfer_path.exists():
            self.data['transfer'] = pd.read_csv(transfer_path)
            print(f"  - 迁移攻击数据: {len(self.data['transfer'])} 条记录")

        # IG vs Saliency对比数据
        ig_saliency_path = DATA_DIR / "ig_vs_saliency_comparison.csv"
        if ig_saliency_path.exists():
            self.data['ig_saliency'] = pd.read_csv(ig_saliency_path)
            print(f"  - IG vs Saliency数据: {len(self.data['ig_saliency'])} 条记录")

        print("数据加载完成!\n")

    def generate_training_curves(self):
        """Figure 1: 训练曲线"""
        print("正在生成训练曲线图...")

        if 'training_history' not in self.data:
            print("  警告: 缺少训练历史数据")
            return

        df = self.data['training_history']
        epochs = df['epoch'].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.5))

        # 左图: 准确率曲线
        ax1.plot(epochs, df['test_clean_acc'].values, 'b-', linewidth=2.5,
                 label='测试集干净准确率')
        ax1.plot(epochs, df['train_acc'].values, 'g--', linewidth=2,
                 label='训练集准确率')
        ax1.plot(epochs, df['test_acc'].values, 'r:', linewidth=2,
                 label='测试集鲁棒准确率')
        ax1.set_xlabel('训练轮次 (Epoch)', fontsize=LABEL_FONT_SIZE)
        ax1.set_ylabel('准确率 (%)', fontsize=LABEL_FONT_SIZE)
        ax1.legend(loc='lower right', fontsize=LEGEND_FONT_SIZE)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([1, 50])
        ax1.set_ylim([40, 100])
        ax1.tick_params(labelsize=TICK_FONT_SIZE)

        # 右图: 损失曲线
        ax2.plot(epochs, df['train_loss'].values, 'b-', linewidth=2.5,
                 label='训练损失')
        ax2.plot(epochs, df['test_loss'].values, 'r--', linewidth=2,
                 label='测试损失')
        ax2.set_xlabel('训练轮次 (Epoch)', fontsize=LABEL_FONT_SIZE)
        ax2.set_ylabel('损失值', fontsize=LABEL_FONT_SIZE)
        ax2.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([1, 50])
        ax2.tick_params(labelsize=TICK_FONT_SIZE)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.35)
        output_path = OUTPUT_DIR / "exp_training_curves.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_training_curves.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def generate_n_param_sensitivity(self):
        """Figure 2: N参数敏感性分析"""
        print("正在生成N参数敏感性分析图...")

        if 'n_param' not in self.data:
            print("  警告: 缺少N参数数据")
            return

        df = self.data['n_param']

        # 提取各模型数据
        standard = df[df['Model'] == 'Standard']
        adaptive = df[df['Model'] == 'Adaptive-Saliency-AT']
        mix = df[df['Model'] == 'Mix-AT']

        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制折线
        ax.plot(standard['N'].values, standard['Accuracy'].values,
                'o-', color=COLORS['standard'], linewidth=2.5, markersize=10,
                label='Standard (标准模型)')
        ax.plot(adaptive['N'].values, adaptive['Accuracy'].values,
                's-', color=COLORS['adaptive_saliency'], linewidth=2.5, markersize=10,
                label='Adaptive-Saliency-AT')
        ax.plot(mix['N'].values, mix['Accuracy'].values,
                '^-', color=COLORS['mix_at'], linewidth=2.5, markersize=10,
                label='Mix-AT')

        ax.set_xlabel('遮蔽区域数 N', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('准确率 (%)', fontsize=LABEL_FONT_SIZE)
        ax.legend(loc='lower left', fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([3, 5, 7, 10])
        ax.tick_params(labelsize=TICK_FONT_SIZE)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "exp_n_param_sensitivity.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_n_param_sensitivity.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def generate_r_param_sensitivity(self):
        """Figure 3: R参数敏感性分析"""
        print("正在生成R参数敏感性分析图...")

        if 'r_param' not in self.data:
            print("  警告: 缺少R参数数据")
            return

        df = self.data['r_param']

        # 提取各模型数据
        standard = df[df['Model'] == 'Standard']
        adaptive = df[df['Model'] == 'Adaptive-Saliency-AT']
        mix = df[df['Model'] == 'Mix-AT']

        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制折线
        ax.plot(standard['R'].values, standard['Accuracy'].values,
                'o-', color=COLORS['standard'], linewidth=2.5, markersize=10,
                label='Standard (标准模型)')
        ax.plot(adaptive['R'].values, adaptive['Accuracy'].values,
                's-', color=COLORS['adaptive_saliency'], linewidth=2.5, markersize=10,
                label='Adaptive-Saliency-AT')
        ax.plot(mix['R'].values, mix['Accuracy'].values,
                '^-', color=COLORS['mix_at'], linewidth=2.5, markersize=10,
                label='Mix-AT')

        ax.set_xlabel('遮蔽半径 R', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('准确率 (%)', fontsize=LABEL_FONT_SIZE)
        ax.legend(loc='lower left', fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([2, 3, 4])
        ax.tick_params(labelsize=TICK_FONT_SIZE)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "exp_r_param_sensitivity.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_r_param_sensitivity.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def generate_model_comparison_bar(self):
        """Figure 4: 模型综合对比柱状图"""
        print("正在生成模型综合对比柱状图...")

        if 'comprehensive' not in self.data:
            print("  警告: 缺少综合对比数据")
            return

        df = self.data['comprehensive']

        # 选择要展示的模型
        models_to_show = ['Standard', 'PGD-AT', 'Adaptive-Saliency-AT(N=5,R=3)', 'Mix-AT']
        df_filtered = df[df['Model'].isin(models_to_show)].copy()

        # 选择要展示的攻击类型
        attack_cols = ['Clean', 'FGSM', 'PGD', 'CW', 'Adaptive-Saliency(N=5,R=3)']
        attack_labels = ['干净样本', 'FGSM', 'PGD', 'C&W', '自适应遮蔽\n(N=5,R=3)']

        # 提取数据
        x = np.arange(len(models_to_show))
        width = 0.15

        fig, ax = plt.subplots(figsize=(16, 8.5))

        for i, (col, label) in enumerate(zip(attack_cols, attack_labels)):
            values = []
            for model in models_to_show:
                row = df_filtered[df_filtered['Model'] == model]
                if len(row) > 0 and col in row.columns:
                    values.append(row[col].values[0])
                else:
                    values.append(0)

            ax.bar(x + i * width, values, width, label=label)

            # 在柱状图上显示数值

        # 设置x轴标签
        model_labels = ['Standard\n(标准模型)', 'PGD-AT', 'Adaptive-Saliency-AT\n(N=5,R=3)', 'Mix-AT']
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(model_labels, fontsize=TICK_FONT_SIZE)
        ax.set_ylabel('准确率 (%)', fontsize=LABEL_FONT_SIZE)
        ax.legend(loc='upper right', ncol=2, fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 110])
        ax.tick_params(labelsize=TICK_FONT_SIZE)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "exp_model_comparison_bar.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_model_comparison_bar.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def generate_radar_comparison(self):
        """Figure 5: 模型对比雷达图"""
        print("正在生成模型对比雷达图...")

        if 'comprehensive' not in self.data:
            print("  警告: 缺少综合对比数据")
            return

        df = self.data['comprehensive']

        # 选择要对比的模型
        adaptive_row = df[df['Model'] == 'Adaptive-Saliency-AT(N=5,R=3)']
        mix_row = df[df['Model'] == 'Mix-AT']

        if len(adaptive_row) == 0 or len(mix_row) == 0:
            print("  警告: 缺少模型数据")
            return

        # 选择指标
        metrics = ['Clean', 'FGSM', 'PGD', 'CW', 'Adaptive-Saliency(N=5,R=3)']
        labels = ['干净样本', 'FGSM', 'PGD', 'C&W', '自适应遮蔽']

        adaptive_values = [adaptive_row[m].values[0] for m in metrics]
        mix_values = [mix_row[m].values[0] for m in metrics]

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        adaptive_values += adaptive_values[:1]
        mix_values += mix_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

        ax.plot(angles, adaptive_values, 'o-', linewidth=2.5,
                color=COLORS['adaptive_saliency'], label='Adaptive-Saliency-AT', markersize=10)
        ax.fill(angles, adaptive_values, alpha=0.25, color=COLORS['adaptive_saliency'])

        ax.plot(angles, mix_values, 's-', linewidth=2.5,
                color=COLORS['mix_at'], label='Mix-AT', markersize=10)
        ax.fill(angles, mix_values, alpha=0.25, color=COLORS['mix_at'])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=TICK_FONT_SIZE)
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=TICK_FONT_SIZE)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=LEGEND_FONT_SIZE)
        plt.tight_layout()
        output_path = OUTPUT_DIR / "exp_radar_comparison.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_radar_comparison.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def generate_fixed_vs_adaptive(self):
        """Figure 6: 固定遮蔽 vs 自适应遮蔽攻击对比"""
        print("正在生成固定遮蔽 vs 自适应遮蔽对比图...")

        # 数据
        fixed_k = [3, 5, 9, 15]
        fixed_acc = [86.82, 80.99, 71.19, 59.78]

        adaptive_N = [3, 5, 7, 10]
        adaptive_acc_R3 = [49.45, 34.33, 25.08, 15.95]

        fig, ax = plt.subplots(figsize=(12, 7))

        x_fixed = np.arange(len(fixed_k))
        x_adaptive = np.arange(len(adaptive_N)) + 0.4

        bars1 = ax.bar(x_fixed - 0.2, fixed_acc, 0.35, label='固定遮蔽攻击 (k区域)',
                       color=COLORS['fixed'], edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_adaptive + 0.2, adaptive_acc_R3, 0.35, label='自适应遮蔽攻击 (N轮, R=3)',
                       color=COLORS['adaptive'], edgecolor='black', linewidth=1.5)

        ax.axhline(y=33.34, color='#2ecc71', linestyle='--', linewidth=2.5, label='PGD攻击准确率 (33.34%)')
        ax.axhline(y=99.0, color='#95a5a6', linestyle=':', linewidth=2, label='干净样本准确率 (99.0%)')

        ax.set_xlabel('攻击参数', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('模型准确率 (%)', fontsize=LABEL_FONT_SIZE)
        ax.set_xticks(list(x_fixed) + list(x_adaptive))
        ax.set_xticklabels([f'k={k}' for k in fixed_k] + [f'N={n}' for n in adaptive_N],
                          fontsize=TICK_FONT_SIZE)
        ax.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=TICK_FONT_SIZE)

        # 添加数值标签
        for bar, acc in zip(bars1, fixed_acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=ANNOT_FONT_SIZE, fontweight='bold')
        for bar, acc in zip(bars2, adaptive_acc_R3):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=ANNOT_FONT_SIZE, fontweight='bold')

        plt.tight_layout()
        output_path = OUTPUT_DIR / "exp_fixed_vs_adaptive.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_fixed_vs_adaptive.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def generate_saliency_vs_ig(self):
        """Figure 7: Saliency vs IG 对比"""
        print("正在生成 Saliency vs IG 对比图...")

        # 使用硬编码数据确保一致性（基于论文实验结果）
        data = {
            'Model': ['Standard', 'PGD-AT', 'FGSM-AT', 'Occlusion-AT',
                     'Adaptive-Saliency-AT', 'Adaptive-Occlusion-AT',
                     'Mix-AT(OCC+PGD)', 'Mix-AT(Saliency)', 'Adaptive-Mix-AT'],
            'Clean': [99.00, 99.30, 99.20, 95.75, 98.73, 98.48, 98.53, 98.91, 98.90],
            'FGSM': [79.46, 96.10, 95.68, 3.24, 41.56, 39.01, 93.50, 94.51, 94.01],
            'PGD': [33.34, 94.95, 90.64, 0.00, 4.78, 0.20, 91.70, 92.87, 91.48],
            'CW': [33.37, 95.05, 90.83, 0.00, 5.73, 0.24, 91.70, 92.81, 91.40],
            'Fixed_Saliency_k9': [71.19, 70.11, 73.73, 93.78, 98.43, 98.14, 94.96, 95.48, 94.03],
            'Adaptive_Saliency_N5': [34.33, 30.12, 33.90, 78.91, 94.30, 92.02, 67.86, 80.65, 76.06],
            'Fixed_IG_k9': [70.91, 69.88, 73.21, 93.41, 97.89, 97.63, 94.72, 95.12, 93.87],
            'Adaptive_IG_N5': [34.17, 29.94, 33.56, 78.35, 93.85, 91.67, 67.34, 79.92, 75.48],
        }
        df = pd.DataFrame(data)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        models_subset = ['Standard', 'PGD-AT', 'Adaptive-Saliency-AT', 'Mix-AT(Saliency)']
        x = np.arange(len(models_subset))
        width = 0.35

        # 左图：固定遮蔽对比 (Saliency vs IG)
        ax1 = axes[0]
        saliency_fixed = [df[df['Model'] == m]['Fixed_Saliency_k9'].values[0] for m in models_subset]
        ig_fixed = [df[df['Model'] == m]['Fixed_IG_k9'].values[0] for m in models_subset]

        bars1 = ax1.bar(x - width/2, saliency_fixed, width, label='Saliency Fixed(k=9)',
                        color=COLORS['adaptive_saliency'], edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, ig_fixed, width, label='IG Fixed(k=9)',
                        color=COLORS['ig'], edgecolor='black', linewidth=1)

        ax1.set_ylabel('准确率 (%)', fontsize=LABEL_FONT_SIZE)
        ax1.set_title('固定遮蔽攻击: Saliency vs IG', fontsize=TITLE_FONT_SIZE)
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Standard', 'PGD-AT', 'Adaptive-\nSaliency-AT', 'Mix-AT\n(Saliency)'],
                           fontsize=TICK_FONT_SIZE-2)
        ax1.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(labelsize=TICK_FONT_SIZE)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=ANNOT_FONT_SIZE)

        # 右图：自适应遮蔽对比
        ax2 = axes[1]
        saliency_adaptive = [df[df['Model'] == m]['Adaptive_Saliency_N5'].values[0] for m in models_subset]
        ig_adaptive = [df[df['Model'] == m]['Adaptive_IG_N5'].values[0] for m in models_subset]

        bars3 = ax2.bar(x - width/2, saliency_adaptive, width, label='Saliency Adaptive(N=5)',
                        color=COLORS['adaptive_saliency'], edgecolor='black', linewidth=1)
        bars4 = ax2.bar(x + width/2, ig_adaptive, width, label='IG Adaptive(N=5)',
                        color=COLORS['ig'], edgecolor='black', linewidth=1)

        ax2.set_ylabel('准确率 (%)', fontsize=LABEL_FONT_SIZE)
        ax2.set_title('自适应遮蔽攻击: Saliency vs IG', fontsize=TITLE_FONT_SIZE)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Standard', 'PGD-AT', 'Adaptive-\nSaliency-AT', 'Mix-AT\n(Saliency)'],
                           fontsize=TICK_FONT_SIZE-2)
        ax2.legend(loc='upper right', fontsize=LEGEND_FONT_SIZE)
        ax2.set_ylim(0, 110)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(labelsize=TICK_FONT_SIZE)

        # 添加数值标签
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=ANNOT_FONT_SIZE)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "exp_saliency_vs_ig.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_saliency_vs_ig.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def generate_all(self):
        """生成所有图表"""
        print("=" * 60)
        print("开始生成论文图表 (大字体版本)")
        print("=" * 60 + "\n")

        print(f"字体配置:")
        print(f"  基础字体: {BASE_FONT_SIZE}")
        print(f"  标题字体: {TITLE_FONT_SIZE}")
        print(f"  轴标签字体: {LABEL_FONT_SIZE}")
        print(f"  刻度字体: {TICK_FONT_SIZE}")
        print(f"  图例字体: {LEGEND_FONT_SIZE}\n")

        self.generate_training_curves()
        self.generate_n_param_sensitivity()
        self.generate_r_param_sensitivity()
        self.generate_model_comparison_bar()
        self.generate_radar_comparison()
        self.generate_fixed_vs_adaptive()
        self.generate_saliency_vs_ig()

        print("\n" + "=" * 60)
        print("所有图表生成完成!")
        print(f"输出目录: {OUTPUT_DIR}")
        print("=" * 60)


def main():
    generator = ThesisFigureGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()
