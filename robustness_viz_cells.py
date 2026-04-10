"""
15.4 鲁棒性评测可视化 - Notebook单元格代码
复制这些代码块到Jupyter Notebook中运行
"""

# ========== Cell 1: 导入和模型加载 ==========
"""
# 由于内核重启，重新加载必要的库和模型
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

# 加载模型函数
def load_model_from_ckpt(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"警告: 模型文件不存在 {ckpt_path}")
        return None
    net = LeNet5().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state['net'])
    net.eval()
    print(f"已加载: {ckpt_path}")
    return net

# 加载各种模型
std_lenet = load_model_from_ckpt('./save_model/50epoch/mnist_lenet5.pth')
cnn = load_model_from_ckpt('./save_model/50epoch/mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth')
cnn_mix = load_model_from_ckpt('./save_model/10epoch/mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth')

# 加载测试数据
imgs, lbls = load_mnist_test()
print(f"测试集大小: {len(imgs)}")
"""

# ========== Cell 2: 辅助函数 ==========
"""
def imshow_with_pred(img, model, true_label, ax=None, title_prefix=''):
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

# 选择每个数字一个样本
sample_indices = []
shown = set()
for i in range(len(lbls)):
    label = int(lbls[i].item())
    if label not in shown:
        shown.add(label)
        sample_indices.append(i)
    if len(shown) == 10:
        break

print(f'样本索引: {sample_indices}')
print(f'样本标签: {[int(lbls[i].item()) for i in sample_indices]}')
"""

# ========== Cell 3: 15.4.1 自适应遮蔽攻击可视化 (不同N, 固定R=3) ==========
"""
# 使用标准模型进行可视化
model = std_lenet
model_name = 'Standard'
N_values = [3, 5, 7, 10]
R_fixed = 3

fig, axes = plt.subplots(10, len(N_values)+1, figsize=(15, 30))

for row, idx in enumerate(sample_indices):
    x = imgs[idx:idx+1].to(device)
    y = lbls[idx:idx+1].to(device)
    true_label = int(y.item())

    # 干净样本
    imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

    # 不同N参数的自适应遮蔽攻击
    for col, N_val in enumerate(N_values):
        attack = AdaptiveSaliencyOcclusionAttack(model, N=N_val, R=R_fixed, c=0.0)
        x_adv = attack((x, y))
        imshow_with_pred(x_adv.squeeze(0), model, true_label,
                        ax=axes[row, col+1], title_prefix=f'Adaptive N={N_val}')

plt.tight_layout()
plt.savefig(f'./results_figures/saliency_Adaptive_attack_visualization_N_{model_name}_R{R_fixed}.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(f'自适应遮蔽攻击(N参数)可视化已保存')
"""

# ========== Cell 4: 15.4.1 自适应遮蔽攻击可视化 (不同R, 固定N=5) ==========
"""
model = std_lenet
model_name = 'Standard'
N_fixed = 5
R_values = [2, 3, 4]

fig, axes = plt.subplots(10, len(R_values)+1, figsize=(12, 30))

for row, idx in enumerate(sample_indices):
    x = imgs[idx:idx+1].to(device)
    y = lbls[idx:idx+1].to(device)
    true_label = int(y.item())

    # 干净样本
    imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

    # 不同R参数的自适应遮蔽攻击
    for col, R_val in enumerate(R_values):
        attack = AdaptiveSaliencyOcclusionAttack(model, N=N_fixed, R=R_val, c=0.0)
        x_adv = attack((x, y))
        imshow_with_pred(x_adv.squeeze(0), model, true_label,
                        ax=axes[row, col+1], title_prefix=f'Adaptive R={R_val}')

plt.tight_layout()
plt.savefig(f'./results_figures/saliency_Adaptive_attack_visualization_R_{model_name}_N{N_fixed}.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(f'自适应遮蔽攻击(R参数)可视化已保存')
"""

# ========== Cell 5: 15.4.2 固定遮蔽攻击可视化 (不同top_k) ==========
"""
model = std_lenet
model_name = 'Standard'
top_k_values = [3, 5, 7, 9, 12, 15]
kernel_size = 3

fig, axes = plt.subplots(10, len(top_k_values)+1, figsize=(18, 30))

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
plt.savefig(f'./results_figures/saliency_fixed_attack_visualization_{model_name}_ks{kernel_size}.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(f'固定遮蔽攻击可视化已保存')
"""

# ========== Cell 6: 15.4.3 不同遮蔽颜色可视化 ==========
"""
model = std_lenet
model_name = 'Standard'
colors = [0.0, 0.5, 1.0]
color_names = ['黑色', '灰色', '白色']
N_color = 5
R_color = 3

fig, axes = plt.subplots(10, len(colors)+1, figsize=(12, 30))

for row, idx in enumerate(sample_indices):
    x = imgs[idx:idx+1].to(device)
    y = lbls[idx:idx+1].to(device)
    true_label = int(y.item())

    # 干净样本
    imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

    # 不同颜色的自适应遮蔽攻击
    for col, (c, name) in enumerate(zip(colors, color_names)):
        attack = AdaptiveSaliencyOcclusionAttack(model, N=N_color, R=R_color, c=c)
        x_adv = attack((x, y))
        imshow_with_pred(x_adv.squeeze(0), model, true_label,
                        ax=axes[row, col+1], title_prefix=f'颜色={name}')

plt.tight_layout()
plt.savefig(f'./results_figures/saliency_color_attack_visualization_{model_name}.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(f'遮蔽颜色攻击可视化已保存')
"""

# ========== Cell 7: PGD攻击可视化 ==========
"""
model = std_lenet
model_name = 'Standard'
eps_values = [0.05, 0.1, 0.15, 0.2]

fig, axes = plt.subplots(10, len(eps_values)+1, figsize=(15, 30))

for row, idx in enumerate(sample_indices):
    x = imgs[idx:idx+1].to(device)
    y = lbls[idx:idx+1].to(device)
    true_label = int(y.item())

    # 干净样本
    imshow_with_pred(x.squeeze(0), model, true_label, ax=axes[row, 0], title_prefix='干净样本')

    # 不同epsilon的PGD攻击
    for col, eps in enumerate(eps_values):
        pgd = LinfPGD(net=model, eps=eps, step=20, step_size=eps/10, random_start=True)
        x_adv = pgd((x, y))
        imshow_with_pred(x_adv.squeeze(0), model, true_label,
                        ax=axes[row, col+1], title_prefix=f'PGD ε={eps}')

plt.tight_layout()
plt.savefig(f'./results_figures/pgd_attack_visualization_{model_name}.png',
            dpi=150, bbox_inches='tight')
plt.show()
print(f'PGD攻击可视化已保存')
"""

# ========== Cell 8: 综合攻击对比可视化 ==========
"""
model = std_lenet  # 可以更换为 cnn 或 cnn_mix
model_name = 'Standard'

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
plt.savefig(f'./results_figures/all_attacks_comparison_{model_name}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'综合攻击对比可视化已保存')
"""

# ========== Cell 9: 多模型对比可视化 (同一个攻击，不同模型) ==========
"""
# 对比标准模型、Adaptive-AT模型、Mix-AT模型在同一攻击下的表现
models = {
    'Standard': std_lenet,
    'Adaptive-AT': cnn,
    'Mix-AT': cnn_mix
}

# 使用Adaptive攻击进行对比
N_test = 5
R_test = 3

fig, axes = plt.subplots(10, len(models)+1, figsize=(12, 30))

for row, idx in enumerate(sample_indices):
    x = imgs[idx:idx+1].to(device)
    y = lbls[idx:idx+1].to(device)
    true_label = int(y.item())

    # 干净样本 (使用标准模型预测)
    imshow_with_pred(x.squeeze(0), std_lenet, true_label, ax=axes[row, 0], title_prefix='干净样本')

    # 不同模型的攻击效果
    for col, (model_name, model) in enumerate(models.items()):
        if model is not None:
            attack = AdaptiveSaliencyOcclusionAttack(model, N=N_test, R=R_test, c=0.0)
            x_adv = attack((x, y))
            imshow_with_pred(x_adv.squeeze(0), model, true_label,
                            ax=axes[row, col+1], title_prefix=f'{model_name}')
        else:
            axes[row, col+1].text(0.5, 0.5, '模型未加载', ha='center', va='center')
            axes[row, col+1].axis('off')

plt.tight_layout()
plt.savefig(f'./results_figures/model_comparison_Adaptive_N{N_test}_R{R_test}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'多模型对比可视化已保存')
"""
