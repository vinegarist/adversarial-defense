# ========== Adaptive-Saliency-PGD-Mixed-AT 迁移攻击测试 ==========
"""
对 save_model\50epoch\mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth 进行迁移攻击测试
使用标准模型作为替代模型生成对抗样本
"""

import sys
sys.path.insert(0, r'D:\软件\对抗性防御\对抗性防御-1\03.代码')

import torch
import torch.nn as nn
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

print(f'Device: {device}')

# ========== 1. 加载模型 ==========
print('\n=== 加载模型 ===')

# 替代模型（标准模型）
std_lenet = LeNet5().to(device)
std_state = torch.load('./save_model/50epoch/mnist_lenet5.pth', map_location=device, weights_only=False)
std_lenet.load_state_dict(std_state['net'])
std_lenet.eval()
print('[OK] 替代模型: Standard')

# 目标模型（AdaptiveSaliencyPgdMixedAT）
cnn_mix = LeNet5().to(device)
mix_state = torch.load('./save_model/50epoch/mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth',
                       map_location=device, weights_only=False)
cnn_mix.load_state_dict(mix_state['net'])
cnn_mix.eval()
print('[OK] 目标模型: Adaptive-Saliency-PGD-Mixed-AT')

# 加载测试数据
print('\n=== 加载测试数据 ===')
imgs, lbls = load_mnist_test()
print(f'测试集大小: {len(imgs)}')

# ========== 2. 迁移攻击测试 ==========
print('\n=== 迁移攻击测试 ===')
print('使用 Standard 模型生成对抗样本，测试 Mixed-AT 模型')
print('-' * 60)

# 攻击参数
N_attack = 5
R_attack = 3
top_k = 9
kernel_size = 3
occlu_color = 0.0
eps = 0.1

results = []

# 1. 干净样本
print('测试干净样本...')
clean_acc, _ = test_fn(cnn_mix, imgs, lbls, bs=250, mode='clean')
results.append({'攻击类型': 'Clean', '准确率': f'{clean_acc:.2f}%'})
print(f'  Clean: {clean_acc:.2f}%')

# 2. FGSM 迁移攻击
print('FGSM 迁移攻击...')
fgsm = LinfPGD(net=std_lenet, eps=eps, step=1, step_size=eps, random_start=False)
fgsm_acc, _ = test_fn(nn.Sequential(fgsm, cnn_mix), imgs, lbls, bs=250, mode='attack')
results.append({'攻击类型': 'FGSM', '准确率': f'{fgsm_acc:.2f}%'})
print(f'  FGSM: {fgsm_acc:.2f}%')

# 3. PGD 迁移攻击
print('PGD 迁移攻击...')
pgd = LinfPGD(net=std_lenet, eps=eps, step=20, step_size=0.025, random_start=True)
pgd_acc, _ = test_fn(nn.Sequential(pgd, cnn_mix), imgs, lbls, bs=250, mode='attack')
results.append({'攻击类型': 'PGD', '准确率': f'{pgd_acc:.2f}%'})
print(f'  PGD: {pgd_acc:.2f}%')

# 4. CW 迁移攻击
print('CW 迁移攻击...')
cw = LinfPGD(net=std_lenet, eps=eps, step=20, step_size=0.025, random_start=True, criterion=CWLoss)
cw_acc, _ = test_fn(nn.Sequential(cw, cnn_mix), imgs, lbls, bs=250, mode='attack')
results.append({'攻击类型': 'CW', '准确率': f'{cw_acc:.2f}%'})
print(f'  CW: {cw_acc:.2f}%')

# 5. Fixed-Saliency 迁移攻击
print('Fixed-Saliency 迁移攻击...')
fixed = SaliencyOcclusionAttack(std_lenet, top_k=top_k, kernel_size=kernel_size, occlu_color=occlu_color)
fixed_acc, _ = test_fn(nn.Sequential(fixed, cnn_mix), imgs, lbls, bs=250, mode='attack')
results.append({'攻击类型': 'Fixed-Saliency', '准确率': f'{fixed_acc:.2f}%'})
print(f'  Fixed-Saliency: {fixed_acc:.2f}%')

# 6. Adaptive-Saliency 迁移攻击
print('Adaptive-Saliency 迁移攻击...')
adaptive = AdaptiveSaliencyOcclusionAttack(std_lenet, N=N_attack, R=R_attack, c=occlu_color)
adaptive_acc, _ = test_fn(nn.Sequential(adaptive, cnn_mix), imgs, lbls, bs=250, mode='attack')
results.append({'攻击类型': 'Adaptive-Saliency', '准确率': f'{adaptive_acc:.2f}%'})
print(f'  Adaptive-Saliency: {adaptive_acc:.2f}%')

# ========== 3. 显示结果表格 ==========
print('\n' + '=' * 70)
print('迁移攻击结果汇总')
print('目标模型: Adaptive-Saliency-PGD-Mixed-AT')
print('替代模型: Standard')
print('=' * 70)

df_result = pd.DataFrame(results)
print(df_result.to_string(index=False))

print('=' * 70)

# 保存CSV
os.makedirs('./results_figures', exist_ok=True)
csv_path = './results_figures/transfer_attack_mixed_at_results.csv'
df_result.to_csv(csv_path, index=False)
print(f'[SAVED] 结果已保存: {csv_path}')
