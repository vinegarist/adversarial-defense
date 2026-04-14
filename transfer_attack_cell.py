# ========== 迁移攻击测试 (使用 AdaptiveSaliencyOcclusionAttack) ==========
"""
使用标准模型作为替代模型生成对抗样本，测试遮蔽攻击 AT 模型的鲁棒性
包含:
1. 加载替代模型（标准模型）和目标模型（AT模型）
2. 使用 AdaptiveSaliencyOcclusionAttack 进行迁移攻击
3. 同时对比 FGSM/PGD/CW/Fixed-Saliency 迁移攻击
4. 对抗样本可视化
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
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

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
    print(f'[OK] 加载 {model_name}: {os.path.basename(ckpt_path)}')
    return net

print('\n=== 加载替代模型（标准模型）===')
std_lenet = load_model('./save_model/50epoch/mnist_lenet5.pth', '替代模型-Standard')

print('\n=== 加载目标模型（遮蔽攻击AT模型）===')
# Adaptive-Saliency-AT 模型 (N=5, R=3)
cnn_at = load_model('./save_model/50epoch/mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth', '目标模型-Adaptive-Saliency-AT')

# 其他AT模型（可选）
cnn_pgd = load_model('./save_model/50epoch/mnist_lenet5_PGD_0.1_5_AT.pth', '目标模型-PGD-AT')
cnn_fgsm = load_model('./save_model/50epoch/mnist_lenet5_FGSM_AT.pth', '目标模型-FGSM-AT')

# 加载测试数据
print('\n=== 加载测试数据 ===')
imgs, lbls = load_mnist_test()
print(f'测试集大小: {len(imgs)}')

# ========== 2. 迁移攻击测试 ==========
print('\n=== 迁移攻击测试 ===')
print('使用 替代模型(Standard) 生成对抗样本，测试 目标模型(AT) 的鲁棒性')
print('-' * 60)

# 攻击参数
N_attack = 5      # Adaptive遮蔽数量
R_attack = 3      # Adaptive遮蔽半径
top_k = 9         # Fixed遮蔽top_k
kernel_size = 3   # Fixed遮蔽kernel_size
occlu_color = 0.0 # 遮蔽颜色（黑色）
eps = 0.1         # PGD/FGSM epsilon

# 测试的目标模型列表
target_models = [
    (cnn_at, 'Adaptive-Saliency-AT'),
    (cnn_pgd, 'PGD-AT'),
    (cnn_fgsm, 'FGSM-AT'),
]

# 只保留成功加载的模型
target_models = [(m, n) for m, n in target_models if m is not None]

transfer_results = []

for target_model, target_name in target_models:
    print(f'\n>>> 测试目标模型: {target_name}')
    result = {'target_model': target_name}

    # 1. 干净样本准确率
    clean_acc, _ = test_fn(target_model, imgs, lbls, bs=250, mode='clean')
    result['Clean'] = clean_acc
    print(f'    Clean: {clean_acc:6.2f}%')

    # 2. FGSM 迁移攻击
    fgsm = LinfPGD(net=std_lenet, eps=eps, step=1, step_size=eps, random_start=False)
    fgsm_acc, _ = test_fn(nn.Sequential(fgsm, target_model), imgs, lbls, bs=250, mode='attack')
    result['FGSM'] = fgsm_acc
    print(f'    FGSM:  {fgsm_acc:6.2f}%')

    # 3. PGD 迁移攻击
    pgd = LinfPGD(net=std_lenet, eps=eps, step=20, step_size=0.025, random_start=True)
    pgd_acc, _ = test_fn(nn.Sequential(pgd, target_model), imgs, lbls, bs=250, mode='attack')
    result['PGD'] = pgd_acc
    print(f'    PGD:   {pgd_acc:6.2f}%')

    # 4. CW 迁移攻击
    cw = LinfPGD(net=std_lenet, eps=eps, step=20, step_size=0.025, random_start=True, criterion=CWLoss)
    cw_acc, _ = test_fn(nn.Sequential(cw, target_model), imgs, lbls, bs=250, mode='attack')
    result['CW'] = cw_acc
    print(f'    CW:    {cw_acc:6.2f}%')

    # 5. Fixed-Saliency 迁移攻击
    fixed_saliency = SaliencyOcclusionAttack(std_lenet, top_k=top_k, kernel_size=kernel_size, occlu_color=occlu_color)
    fixed_acc, _ = test_fn(nn.Sequential(fixed_saliency, target_model), imgs, lbls, bs=250, mode='attack')
    result['Fixed-Saliency'] = fixed_acc
    print(f'    Fixed: {fixed_acc:6.2f}%')

    # 6. Adaptive-Saliency 迁移攻击
    adaptive_saliency = AdaptiveSaliencyOcclusionAttack(std_lenet, N=N_attack, R=R_attack, c=occlu_color)
    adaptive_acc, _ = test_fn(nn.Sequential(adaptive_saliency, target_model), imgs, lbls, bs=250, mode='attack')
    result['Adaptive-Saliency'] = adaptive_acc
    print(f'    Adapt: {adaptive_acc:6.2f}%')

    transfer_results.append(result)

# 打印汇总表格
print('\n' + '=' * 70)
print('迁移攻击结果汇总（使用 Standard 模型生成对抗样本）')
print('=' * 70)
import pandas as pd
df_transfer = pd.DataFrame(transfer_results)
print(df_transfer.to_string(index=False))
print('=' * 70)

# ========== 3. 对抗样本可视化 ==========
print('\n=== 迁移攻击对抗样本可视化 ===')

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

# 辅助函数
def imshow_with_pred(img, model, true_label, ax=None, title_prefix=''):
    """显示图像并标注预测结果"""
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

# 目标模型用于可视化（使用第一个可用模型）
if target_models:
    viz_model = target_models[0][0]
    viz_model_name = target_models[0][1]

    print(f'\n使用目标模型 {viz_model_name} 进行可视化')

    # 创建攻击器
    attacks = {
        'Clean': None,
        'FGSM': LinfPGD(net=std_lenet, eps=eps, step=1, step_size=eps, random_start=False),
        'PGD': LinfPGD(net=std_lenet, eps=eps, step=20, step_size=0.025, random_start=True),
        'CW': LinfPGD(net=std_lenet, eps=eps, step=20, step_size=0.025, random_start=True, criterion=CWLoss),
        'Fixed-Saliency': SaliencyOcclusionAttack(std_lenet, top_k=top_k, kernel_size=kernel_size, occlu_color=occlu_color),
        'Adaptive-Saliency': AdaptiveSaliencyOcclusionAttack(std_lenet, N=N_attack, R=R_attack, c=occlu_color),
    }

    # 创建大图
    fig, axes = plt.subplots(10, 7, figsize=(18, 30))

    for row, idx in enumerate(sample_indices):
        x = imgs[idx:idx+1].to(device)
        y = lbls[idx:idx+1].to(device)
        true_label = int(y.item())

        for col, (attack_name, attack) in enumerate(attacks.items()):
            if attack is None:
                # 干净样本
                imshow_with_pred(x.squeeze(0), viz_model, true_label,
                               ax=axes[row, col], title_prefix=f'{attack_name}')
            else:
                # 对抗样本
                x_adv = attack((x, y))
                imshow_with_pred(x_adv.squeeze(0), viz_model, true_label,
                               ax=axes[row, col], title_prefix=f'{attack_name}')

    plt.tight_layout()
    save_path = './results_figures/transfer_attack_visualization_AdaptiveSaliency.png'
    os.makedirs('./results_figures', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\n[SAVED] 可视化结果: {save_path}')
else:
    print('[WARN] 没有可用的目标模型，跳过可视化')

print('\n========== 迁移攻击测试完成 ==========')
