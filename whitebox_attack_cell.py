# ========== 白盒攻击测试 + 迁移攻击对比 ==========
"""
白盒攻击测试：使用 AdaptiveSaliencyOcclusionAttack 等攻击直接攻击目标模型
并与迁移攻击结果汇总对比
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

from occlusion_attack import SaliencyOcclusionAttack, AdaptiveSaliencyOcclusionAttack
from pgd import LinfPGD
from loss import CWLoss
from models import LeNet5
from utils import load_mnist_test
import test
test_fn = test.test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

print(f'Device: {device}')

# ========== 1. 加载所有模型 ==========
def load_model(ckpt_path, model_name='Model'):
    """加载模型"""
    if not os.path.exists(ckpt_path):
        print(f'[WARN] 模型不存在: {ckpt_path}')
        return None
    net = LeNet5().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(state['net'])
    net.eval()
    print(f'[OK] 加载 {model_name}')
    return net

print('\n' + '='*60)
print('加载所有模型')
print('='*60)

# 标准模型（用于迁移攻击生成对抗样本）
std_lenet = load_model('./save_model/50epoch/mnist_lenet5.pth', 'Standard (替代模型)')

# 各AT模型（白盒攻击目标 + 迁移攻击目标）
target_models = {
    'Standard': std_lenet,
    'PGD-AT': load_model('./save_model/50epoch/mnist_lenet5_PGD_0.1_5_AT.pth', 'PGD-AT'),
    'FGSM-AT': load_model('./save_model/50epoch/mnist_lenet5_FGSM_AT.pth', 'FGSM-AT'),
    'Occlusion-AT': load_model('./save_model/50epoch/mnist_lenet5_OcclusionAT_9_3.pth', 'Occlusion-AT'),
    'Adaptive-Occlusion-AT': load_model('./save_model/50epoch/mnist_lenet5_AdaptiveOcclusionAT_5_3.pth', 'Adaptive-Occlusion-AT'),
    'Adaptive-Saliency-AT': load_model('./save_model/50epoch/mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth', 'Adaptive-Saliency-AT'),
}

# 过滤未加载成功的模型
target_models = {k: v for k, v in target_models.items() if v is not None}

if not target_models:
    print('[ERROR] 没有成功加载任何模型！')
    raise RuntimeError('模型加载失败')

print(f'\n成功加载 {len(target_models)} 个模型: {list(target_models.keys())}')

# 加载测试数据
print('\n加载测试数据...')
imgs, lbls = load_mnist_test()
print(f'测试集大小: {len(imgs)}')

# ========== 2. 定义攻击参数 ==========
print('\n' + '='*60)
print('攻击参数配置')
print('='*60)

# 通用参数
EPS = 0.1
N_attack = 5      # Adaptive遮蔽数量
R_attack = 3      # Adaptive遮蔽半径
top_k = 9         # Fixed遮蔽top_k
kernel_size = 3   # Fixed遮蔽kernel_size
occlu_color = 0.0 # 遮蔽颜色（黑色）

print(f'PGD/FGSM epsilon: {EPS}')
print(f'Adaptive-Saliency: N={N_attack}, R={R_attack}')
print(f'Fixed-Saliency: top_k={top_k}, kernel_size={kernel_size}')

# ========== 3. 白盒攻击测试 ==========
def run_whitebox_tests(model, model_name):
    """对白盒模型进行攻击测试"""
    print(f'\n>>> 白盒攻击测试: {model_name}')
    results = {'model': model_name, 'attack_type': 'White-box'}

    # Clean
    clean_acc, _ = test_fn(model, imgs, lbls, bs=250, mode='clean')
    results['Clean'] = clean_acc
    print(f'    Clean:              {clean_acc:6.2f}%')

    # FGSM
    fgsm = LinfPGD(net=model, eps=EPS, step=1, step_size=EPS, random_start=False)
    fgsm_acc, _ = test_fn(nn.Sequential(fgsm, model), imgs, lbls, bs=250, mode='attack')
    results['FGSM'] = fgsm_acc
    print(f'    FGSM:               {fgsm_acc:6.2f}%')

    # PGD
    pgd = LinfPGD(net=model, eps=EPS, step=20, step_size=0.025, random_start=True)
    pgd_acc, _ = test_fn(nn.Sequential(pgd, model), imgs, lbls, bs=250, mode='attack')
    results['PGD'] = pgd_acc
    print(f'    PGD:                {pgd_acc:6.2f}%')

    # CW
    cw = LinfPGD(net=model, eps=EPS, step=20, step_size=0.025, random_start=True, criterion=CWLoss)
    cw_acc, _ = test_fn(nn.Sequential(cw, model), imgs, lbls, bs=250, mode='attack')
    results['CW'] = cw_acc
    print(f'    CW:                 {cw_acc:6.2f}%')

    # Fixed-Saliency
    fixed = SaliencyOcclusionAttack(model, top_k=top_k, kernel_size=kernel_size, occlu_color=occlu_color)
    fixed_acc, _ = test_fn(nn.Sequential(fixed, model), imgs, lbls, bs=250, mode='attack')
    results['Fixed-Saliency'] = fixed_acc
    print(f'    Fixed-Saliency:     {fixed_acc:6.2f}%')

    # Adaptive-Saliency (重点)
    adaptive = AdaptiveSaliencyOcclusionAttack(model, N=N_attack, R=R_attack, c=occlu_color)
    adaptive_acc, _ = test_fn(nn.Sequential(adaptive, model), imgs, lbls, bs=250, mode='attack')
    results['Adaptive-Saliency'] = adaptive_acc
    print(f'    Adaptive-Saliency:  {adaptive_acc:6.2f}%')

    return results


print('\n' + '='*60)
print('白盒攻击测试')
print('='*60)

whitebox_results = []
for name, model in target_models.items():
    result = run_whitebox_tests(model, name)
    whitebox_results.append(result)

# ========== 4. 迁移攻击测试 ==========
def run_transfer_tests(target_model, target_name, substitute_model):
    """使用替代模型生成对抗样本，测试目标模型"""
    print(f'\n>>> 迁移攻击测试: {target_name} (使用 {substitute_model[0]} 生成)')
    results = {'model': target_name, 'attack_type': f'Transfer ({substitute_model[0]})'}

    # Clean
    clean_acc, _ = test_fn(target_model, imgs, lbls, bs=250, mode='clean')
    results['Clean'] = clean_acc
    print(f'    Clean:              {clean_acc:6.2f}%')

    # FGSM
    fgsm = LinfPGD(net=substitute_model[1], eps=EPS, step=1, step_size=EPS, random_start=False)
    fgsm_acc, _ = test_fn(nn.Sequential(fgsm, target_model), imgs, lbls, bs=250, mode='attack')
    results['FGSM'] = fgsm_acc
    print(f'    FGSM:               {fgsm_acc:6.2f}%')

    # PGD
    pgd = LinfPGD(net=substitute_model[1], eps=EPS, step=20, step_size=0.025, random_start=True)
    pgd_acc, _ = test_fn(nn.Sequential(pgd, target_model), imgs, lbls, bs=250, mode='attack')
    results['PGD'] = pgd_acc
    print(f'    PGD:                {pgd_acc:6.2f}%')

    # CW
    cw = LinfPGD(net=substitute_model[1], eps=EPS, step=20, step_size=0.025, random_start=True, criterion=CWLoss)
    cw_acc, _ = test_fn(nn.Sequential(cw, target_model), imgs, lbls, bs=250, mode='attack')
    results['CW'] = cw_acc
    print(f'    CW:                 {cw_acc:6.2f}%')

    # Fixed-Saliency
    fixed = SaliencyOcclusionAttack(substitute_model[1], top_k=top_k, kernel_size=kernel_size, occlu_color=occlu_color)
    fixed_acc, _ = test_fn(nn.Sequential(fixed, target_model), imgs, lbls, bs=250, mode='attack')
    results['Fixed-Saliency'] = fixed_acc
    print(f'    Fixed-Saliency:     {fixed_acc:6.2f}%')

    # Adaptive-Saliency
    adaptive = AdaptiveSaliencyOcclusionAttack(substitute_model[1], N=N_attack, R=R_attack, c=occlu_color)
    adaptive_acc, _ = test_fn(nn.Sequential(adaptive, target_model), imgs, lbls, bs=250, mode='attack')
    results['Adaptive-Saliency'] = adaptive_acc
    print(f'    Adaptive-Saliency:  {adaptive_acc:6.2f}%')

    return results


print('\n' + '='*60)
print('迁移攻击测试')
print('='*60)

transfer_results = []
if 'Standard' in target_models and std_lenet is not None:
    substitute = ('Standard', std_lenet)
    for name, model in target_models.items():
        if name != 'Standard':  # 不对标准模型做迁移攻击
            result = run_transfer_tests(model, name, substitute)
            transfer_results.append(result)

# ========== 5. 汇总对比表格 ==========
print('\n' + '='*80)
print('白盒攻击 vs 迁移攻击 汇总对比')
print('='*80)

# 合并所有结果
all_results = whitebox_results + transfer_results

# 创建DataFrame
df_all = pd.DataFrame(all_results)

# 按模型分组显示
for model_name in target_models.keys():
    print(f'\n>>> {model_name}')
    model_data = df_all[df_all['model'] == model_name]
    print(model_data.to_string(index=False))

print('\n' + '='*80)

# 保存完整表格
os.makedirs('./results_figures', exist_ok=True)
csv_path = './results_figures/whitebox_vs_transfer_attack_comparison.csv'
df_all.to_csv(csv_path, index=False)
print(f'[SAVED] 完整对比表格: {csv_path}')

# ========== 6. 可视化对比 ==========
print('\n=== 白盒 vs 迁移攻击 可视化对比 ===')

# 选择对比的模型（排除Standard，因为没有迁移攻击）
compare_models = [m for m in target_models.keys() if m != 'Standard']

if len(compare_models) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    attack_types = ['FGSM', 'PGD', 'CW', 'Fixed-Saliency', 'Adaptive-Saliency']

    for idx, attack in enumerate(attack_types):
        ax = axes[idx // 3, idx % 3]

        x = np.arange(len(compare_models))
        width = 0.35

        whitebox_vals = []
        transfer_vals = []

        for model in compare_models:
            wb_val = df_all[(df_all['model'] == model) & (df_all['attack_type'] == 'White-box')][attack].values
            tr_val = df_all[(df_all['model'] == model) & (df_all['attack_type'].str.startswith('Transfer'))][attack].values

            whitebox_vals.append(wb_val[0] if len(wb_val) > 0 else 0)
            transfer_vals.append(tr_val[0] if len(tr_val) > 0 else 0)

        ax.bar(x - width/2, whitebox_vals, width, label='White-box', color='#1f77b4', alpha=0.8)
        ax.bar(x + width/2, transfer_vals, width, label='Transfer', color='#ff7f0e', alpha=0.8)

        ax.set_xlabel('Target Model', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{attack} Attack', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(compare_models, rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)

        # 添加数值标签
        for i, (wb, tr) in enumerate(zip(whitebox_vals, transfer_vals)):
            ax.text(i - width/2, wb + 2, f'{wb:.1f}', ha='center', fontsize=8)
            ax.text(i + width/2, tr + 2, f'{tr:.1f}', ha='center', fontsize=8)

    plt.tight_layout()
    save_path = './results_figures/whitebox_vs_transfer_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'[SAVED] 对比图: {save_path}')

print('\n' + '='*60)
print('白盒攻击测试 + 迁移攻击对比 完成！')
print('='*60)
