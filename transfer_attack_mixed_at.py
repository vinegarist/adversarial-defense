# ========== AdaptiveSaliencyPgdMixedAT 迁移攻击测试 ==========
"""
对 save_model\50epoch\mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth
使用标准模型作为替代模型进行迁移攻击测试
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

print('\n=== 加载模型 ===')
# 替代模型（标准模型）
std_lenet = load_model('./save_model/50epoch/mnist_lenet5.pth', '替代模型-Standard')

# 目标模型（Mixed-AT）
cnn_mixed = load_model('./save_model/50epoch/mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth',
                       '目标模型-AdaptiveSaliencyPgdMixedAT')

if std_lenet is None or cnn_mixed is None:
    print('[ERROR] 模型加载失败！')
    raise RuntimeError('模型加载失败')

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

results = {'model': 'AdaptiveSaliencyPgdMixedAT', 'attack_type': 'Transfer (Standard)'}

# Clean
clean_acc, _ = test_fn(cnn_mixed, imgs, lbls, bs=250, mode='clean')
results['Clean'] = clean_acc
print(f'Clean:              {clean_acc:6.2f}%')

# FGSM
fgsm = LinfPGD(net=std_lenet, eps=eps, step=1, step_size=eps, random_start=False)
fgsm_acc, _ = test_fn(nn.Sequential(fgsm, cnn_mixed), imgs, lbls, bs=250, mode='attack')
results['FGSM'] = fgsm_acc
print(f'FGSM:               {fgsm_acc:6.2f}%')

# PGD
pgd = LinfPGD(net=std_lenet, eps=eps, step=20, step_size=0.025, random_start=True)
pgd_acc, _ = test_fn(nn.Sequential(pgd, cnn_mixed), imgs, lbls, bs=250, mode='attack')
results['PGD'] = pgd_acc
print(f'PGD:                {pgd_acc:6.2f}%')

# CW
cw = LinfPGD(net=std_lenet, eps=eps, step=20, step_size=0.025, random_start=True, criterion=CWLoss)
cw_acc, _ = test_fn(nn.Sequential(cw, cnn_mixed), imgs, lbls, bs=250, mode='attack')
results['CW'] = cw_acc
print(f'CW:                 {cw_acc:6.2f}%')

# Fixed-Saliency
fixed = SaliencyOcclusionAttack(std_lenet, top_k=top_k, kernel_size=kernel_size, occlu_color=occlu_color)
fixed_acc, _ = test_fn(nn.Sequential(fixed, cnn_mixed), imgs, lbls, bs=250, mode='attack')
results['Fixed-Saliency'] = fixed_acc
print(f'Fixed-Saliency:     {fixed_acc:6.2f}%')

# Adaptive-Saliency
adaptive = AdaptiveSaliencyOcclusionAttack(std_lenet, N=N_attack, R=R_attack, c=occlu_color)
adaptive_acc, _ = test_fn(nn.Sequential(adaptive, cnn_mixed), imgs, lbls, bs=250, mode='attack')
results['Adaptive-Saliency'] = adaptive_acc
print(f'Adaptive-Saliency:  {adaptive_acc:6.2f}%')

# ========== 3. 显示结果表格 ==========
print('\n' + '=' * 100)
print('迁移攻击结果 - AdaptiveSaliencyPgdMixedAT (使用 Standard 替代模型)')
print('=' * 100)

df_result = pd.DataFrame([results])
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print(df_result.to_string(index=False))

print()
print('=' * 100)

# 保存CSV
os.makedirs('./results_figures', exist_ok=True)
csv_path = './results_figures/transfer_attack_mixed_at_result.csv'
df_result.to_csv(csv_path, index=False)
print(f'[SAVED] 结果已保存: {csv_path}')
print('=' * 100)
