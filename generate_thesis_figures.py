# -*- coding: utf-8 -*-
"""
论文图表生成脚本
将 Jupyter notebook 实验成果整理成论文图表

使用方法:
    conda activate adv-attack
    python generate_thesis_figures.py
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

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'

# 颜色配置
COLORS = {
    'standard': '#7F7F7F',
    'pgd_at': '#1f77b4',
    'adaptive_saliency': '#2ca02c',
    'mix_at': '#ff7f0e',
    'highlight': '#d62728'
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

        print("数据加载完成!\n")

    def generate_training_curves(self):
        """Figure 1: 训练曲线"""
        print("正在生成训练曲线图...")

        if 'training_history' not in self.data:
            print("  警告: 缺少训练历史数据")
            return

        df = self.data['training_history']
        epochs = df['epoch'].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 左图: 准确率曲线
        ax1.plot(epochs, df['test_clean_acc'].values, 'b-', linewidth=2,
                 label='测试集干净准确率')
        ax1.plot(epochs, df['train_acc'].values, 'g--', linewidth=1.5,
                 label='训练集准确率')
        ax1.plot(epochs, df['test_acc'].values, 'r:', linewidth=1.5,
                 label='测试集鲁棒准确率')
        ax1.set_xlabel('训练轮次 (Epoch)')
        ax1.set_ylabel('准确率 (%)')
        ax1.set_title('Adaptive-Saliency-AT(N=5,R=3) 准确率曲线')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([1, 50])
        ax1.set_ylim([40, 100])

        # 右图: 损失曲线
        ax2.plot(epochs, df['train_loss'].values, 'b-', linewidth=2,
                 label='训练损失')
        ax2.plot(epochs, df['test_loss'].values, 'r--', linewidth=1.5,
                 label='测试损失')
        ax2.set_xlabel('训练轮次 (Epoch)')
        ax2.set_ylabel('损失值')
        ax2.set_title('Adaptive-Saliency-AT(N=5,R=3) 损失曲线')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([1, 50])

        plt.tight_layout()
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

        fig, ax = plt.subplots(figsize=(8, 6))

        # 绘制折线
        ax.plot(standard['N'].values, standard['Accuracy'].values,
                'o-', color=COLORS['standard'], linewidth=2, markersize=8,
                label='Standard (标准模型)')
        ax.plot(adaptive['N'].values, adaptive['Accuracy'].values,
                's-', color=COLORS['adaptive_saliency'], linewidth=2, markersize=8,
                label='Adaptive-Saliency-AT')
        ax.plot(mix['N'].values, mix['Accuracy'].values,
                '^-', color=COLORS['mix_at'], linewidth=2, markersize=8,
                label='Mix-AT')

        ax.set_xlabel('遮蔽区域数 N')
        ax.set_ylabel('准确率 (%)')
        ax.set_title('N参数敏感性分析 (R=3)')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([3, 5, 7, 10])

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

        fig, ax = plt.subplots(figsize=(8, 6))

        # 绘制折线
        ax.plot(standard['R'].values, standard['Accuracy'].values,
                'o-', color=COLORS['standard'], linewidth=2, markersize=8,
                label='Standard (标准模型)')
        ax.plot(adaptive['R'].values, adaptive['Accuracy'].values,
                's-', color=COLORS['adaptive_saliency'], linewidth=2, markersize=8,
                label='Adaptive-Saliency-AT')
        ax.plot(mix['R'].values, mix['Accuracy'].values,
                '^-', color=COLORS['mix_at'], linewidth=2, markersize=8,
                label='Mix-AT')

        ax.set_xlabel('遮蔽半径 R')
        ax.set_ylabel('准确率 (%)')
        ax.set_title('R参数敏感性分析 (N=5)')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([2, 3, 4])

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

        # 选择要展示的攻击类型（删除Fixed-Saliency）
        attack_cols = ['Clean', 'FGSM', 'PGD', 'CW', 'Adaptive-Saliency(N=5,R=3)']
        attack_labels = ['干净样本', 'FGSM', 'PGD', 'C&W', '自适应遮蔽\n(N=5,R=3)']

        # 提取数据
        x = np.arange(len(models_to_show))
        width = 0.15

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (col, label) in enumerate(zip(attack_cols, attack_labels)):
            values = []
            for model in models_to_show:
                row = df_filtered[df_filtered['Model'] == model]
                if len(row) > 0 and col in row.columns:
                    values.append(row[col].values[0])
                else:
                    values.append(0)

            bars = ax.bar(x + i * width, values, width, label=label)

            # 在柱状图上显示数值
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(f'{val:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)

        # 设置x轴标签
        model_labels = ['Standard\n(标准模型)', 'PGD-AT', 'Adaptive-Saliency-AT\n(N=5,R=3)', 'Mix-AT']
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel('准确率 (%)')
        ax.set_title('各防御策略在不同攻击下的准确率对比')
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 110])

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

        # 选择指标（删除Fixed-Saliency）
        metrics = ['Clean', 'FGSM', 'PGD', 'CW', 'Adaptive-Saliency(N=5,R=3)']
        labels = ['干净样本', 'FGSM', 'PGD', 'C&W', '自适应遮蔽']

        adaptive_values = [adaptive_row[m].values[0] for m in metrics]
        mix_values = [mix_row[m].values[0] for m in metrics]

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        adaptive_values += adaptive_values[:1]
        mix_values += mix_values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, adaptive_values, 'o-', linewidth=2,
                color=COLORS['adaptive_saliency'], label='Adaptive-Saliency-AT')
        ax.fill(angles, adaptive_values, alpha=0.25, color=COLORS['adaptive_saliency'])

        ax.plot(angles, mix_values, 's-', linewidth=2,
                color=COLORS['mix_at'], label='Mix-AT')
        ax.fill(angles, mix_values, alpha=0.25, color=COLORS['mix_at'])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Adaptive-Saliency-AT vs Mix-AT 性能对比', y=1.08)

        plt.tight_layout()
        output_path = OUTPUT_DIR / "exp_radar_comparison.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_radar_comparison.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def generate_mask_visualization(self):
        """Figure 6: 遮蔽效果可视化"""
        print("正在生成遮蔽效果可视化图...")

        # 加载模型和数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载MNIST数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_dataset = datasets.MNIST(
            root=r'd:\软件\对抗性防御\对抗性防御-1\03.代码\data',
            train=False, download=True, transform=transform
        )

        # 选择几个样本进行可视化
        sample_indices = [0, 1, 2, 3, 4]

        fig, axes = plt.subplots(3, len(sample_indices), figsize=(15, 9))

        for i, idx in enumerate(sample_indices):
            img, label = test_dataset[idx]
            img_tensor = img.unsqueeze(0).to(device)

            # 显示原始图像
            img_display = img.squeeze().numpy()
            axes[0, i].imshow(img_display, cmap='gray')
            axes[0, i].set_title(f'原始样本\n标签: {label}')
            axes[0, i].axis('off')

            # 生成自适应遮蔽 (N=5, R=3)
            mask_n5_r3 = self._generate_adaptive_mask(img_tensor, N=5, R=3, device=device)
            occluded_n5_r3 = img_display * (1 - mask_n5_r3)
            axes[1, i].imshow(occluded_n5_r3, cmap='gray')
            axes[1, i].set_title(f'自适应遮蔽\nN=5, R=3')
            axes[1, i].axis('off')

            # 生成自适应遮蔽 (N=10, R=3)
            mask_n10_r3 = self._generate_adaptive_mask(img_tensor, N=10, R=3, device=device)
            occluded_n10_r3 = img_display * (1 - mask_n10_r3)
            axes[2, i].imshow(occluded_n10_r3, cmap='gray')
            axes[2, i].set_title(f'自适应遮蔽\nN=10, R=3')
            axes[2, i].axis('off')

        plt.suptitle('自适应显著性遮蔽攻击效果可视化', fontsize=14)
        plt.tight_layout()
        output_path = OUTPUT_DIR / "exp_mask_visualization.pdf"
        plt.savefig(output_path, format='pdf')
        plt.savefig(OUTPUT_DIR / "exp_mask_visualization.png", format='png')
        plt.close()
        print(f"  已保存: {output_path}")

    def _generate_adaptive_mask(self, img_tensor, N=5, R=3, device='cpu'):
        """生成自适应遮蔽掩码"""
        # 使用简单的梯度显著性方法
        model = LeNet5().to(device)
        model.eval()

        img_tensor = img_tensor.clone().requires_grad_(True)

        # 前向传播
        output = model(img_tensor)
        pred = output.argmax(dim=1)

        # 计算梯度
        loss = output[0, pred]
        loss.backward()

        # 获取显著性图
        saliency = img_tensor.grad.abs().squeeze().cpu().numpy()

        # 生成遮蔽掩码
        mask = np.zeros_like(saliency)

        # 使用卷积进行邻域聚合
        import torch.nn.functional as F_torch
        kernel_size = 2 * R + 1
        # 使用 PyTorch 实现均值滤波
        saliency_tensor = torch.tensor(saliency).unsqueeze(0).unsqueeze(0).float()
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        saliency_sum = F_torch.conv2d(saliency_tensor, kernel, padding=kernel_size//2).squeeze().numpy()

        # 迭代遮蔽
        for _ in range(N):
            # 找到最大值位置
            max_idx = np.unravel_index(np.argmax(saliency_sum), saliency_sum.shape)
            row, col = max_idx

            # 在掩码上标记遮蔽区域
            r_start = max(0, row - R)
            r_end = min(28, row + R + 1)
            c_start = max(0, col - R)
            c_end = min(28, col + R + 1)

            mask[r_start:r_end, c_start:c_end] = 1

            # 将已遮蔽区域的显著性设为0
            saliency_sum[r_start:r_end, c_start:c_end] = 0

        return mask

    def generate_all(self):
        """生成所有图表"""
        print("=" * 60)
        print("开始生成论文图表")
        print("=" * 60 + "\n")

        self.generate_training_curves()
        self.generate_n_param_sensitivity()
        self.generate_r_param_sensitivity()
        self.generate_model_comparison_bar()
        self.generate_radar_comparison()
        self.generate_mask_visualization()

        print("\n" + "=" * 60)
        print("所有图表生成完成!")
        print(f"输出目录: {OUTPUT_DIR}")
        print("=" * 60)


def main():
    generator = ThesisFigureGenerator()
    generator.generate_all()


if __name__ == "__main__":
    main()
