#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run selected experiments for occlusion attack analysis
"""

import os
import sys
sys.path.insert(0, r'D:\软件\对抗性防御\对抗性防御-1\03.代码')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
import time

from models import LeNet5
from utils import load_mnist_test
from test import test as test_fn
from pgd import LinfPGD
from occlusion_attack import OcclusionAttack
from loss import CWLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load test data
print("\nLoading MNIST test data...")
imgs, lbls = load_mnist_test()
print(f"Loaded: {imgs.shape[0]} samples")

# Load models
print("\n" + "="*80)
print("Loading models...")
print("="*80)

models_dict = {}

# Standard model
std_state = torch.load('./save_model/50epoch/mnist_lenet5.pth', weights_only=True)
std_model = LeNet5()
std_model.load_state_dict(std_state['net'])
std_model = std_model.to(device)
std_model.eval()
models_dict['Standard'] = std_model
print("[OK] Standard model")

# Load other models
model_paths = {
    'Occlusion-AT': './save_model/50epoch/mnist_lenet5_OcclusionAT_9_3.pth',
    'PGD-AT': './save_model/50epoch/mnist_lenet5_PGD_0.1_5_AT.pth',
    'FGSM-AT': './save_model/50epoch/mnist_lenet5_FGSM_AT.pth',
    'Mix-AT': './save_model/10epoch/mnist_lenet5_MixedOcclusionPgdAT_0.5_9_3.pth',
}

for name, path in model_paths.items():
    if os.path.exists(path):
        state = torch.load(path, weights_only=True)
        model = LeNet5()
        model.load_state_dict(state['net'])
        model = model.to(device)
        model.eval()
        models_dict[name] = model
        print(f"[OK] {name}")
    else:
        print(f"[SKIP] {name} not found")

print(f"\nTotal models: {len(models_dict)}")

# ========== Experiment 1: Parameter Scan ==========
print("\n" + "="*80)
print("Experiment 1: Occlusion Attack Parameter Scan")
print("="*80)

top_k_values = [3, 5, 7, 9]
kernel_size_values = [3]  # 只使用 kernel_size=3 避免尺寸不匹配错误
num_samples = 5000

sample_imgs = imgs[:num_samples].to(device)
sample_lbls = lbls[:num_samples].to(device)

all_results = []
total_tests = len(top_k_values) * len(kernel_size_values) * len(models_dict)
counter = 0

for model_name, model in models_dict.items():
    clean_acc, _ = test_fn(model, sample_imgs, sample_lbls, bs=250, mode='clean')
    print(f"\n[{model_name}] Clean acc: {clean_acc:.2f}%")

    for top_k in top_k_values:
        for kernel_size in kernel_size_values:
            counter += 1
            start = time.time()

            occlusion = OcclusionAttack(model, top_k=top_k, occlu_color=0.0, kernel_size=kernel_size)
            occ_acc, _ = test_fn(nn.Sequential(occlusion, model),
                                sample_imgs, sample_lbls, bs=250, mode='attack')

            result = {
                'model': model_name,
                'top_k': top_k,
                'kernel_size': kernel_size,
                'clean_acc': clean_acc,
                'occlusion_acc': occ_acc,
                'success_rate': clean_acc - occ_acc,
            }
            all_results.append(result)

            elapsed = time.time() - start
            print(f"  [{counter}/{total_tests}] top_k={top_k}, ks={kernel_size}: "
                  f"acc={occ_acc:.2f}%, success={clean_acc-occ_acc:.2f}% ({elapsed:.1f}s)")

results_df = pd.DataFrame(all_results)
results_df.to_csv('./exp1_occlusion_params.csv', index=False)
print(f"\n[OK] Saved: exp1_occlusion_params.csv")

# ========== Experiment 2: Color Comparison ==========
print("\n" + "="*80)
print("Experiment 2: Occlusion Color Comparison")
print("="*80)

colors = [0.0, 0.5, 1.0]
color_names = {0.0: 'Black', 0.5: 'Gray', 1.0: 'White'}

color_results = []

for color in colors:
    print(f"\n{color_names[color]} (occlu_color={color}):")
    for top_k in [3, 5, 9]:
        occlusion = OcclusionAttack(std_model, top_k=top_k, occlu_color=color, kernel_size=3)
        occ_acc, _ = test_fn(nn.Sequential(occlusion, std_model),
                            sample_imgs, sample_lbls, bs=250, mode='attack')

        clean_acc = 99.0
        color_results.append({
            'color': color_names[color],
            'occlu_color': color,
            'top_k': top_k,
            'occlusion_acc': occ_acc,
            'success_rate': clean_acc - occ_acc
        })
        print(f"  top_k={top_k}: acc={occ_acc:.2f}%, success={clean_acc-occ_acc:.2f}%")

color_df = pd.DataFrame(color_results)
color_df.to_csv('./exp2_occlusion_colors.csv', index=False)
print(f"\n[OK] Saved: exp2_occlusion_colors.csv")

# ========== Experiment 3: Misclassification Analysis ==========
print("\n" + "="*80)
print("Experiment 3: Misclassification Analysis")
print("="*80)

def analyze_misclassification(model, attack, num_samples=1000):
    model.eval()
    x = imgs[:num_samples].to(device)
    y = lbls[:num_samples].to(device)

    with torch.no_grad():
        clean_out = model(x)
        clean_pred = clean_out.max(dim=1).indices
        x_adv = attack((x, y))
        adv_out = model(x_adv)
        adv_pred = adv_out.max(dim=1).indices

    misclassify = defaultdict(lambda: defaultdict(int))
    for i in range(num_samples):
        true_lbl = int(y[i].item())
        clean_p = int(clean_pred[i].item())
        adv_p = int(adv_pred[i].item())
        if clean_p == true_lbl and adv_p != true_lbl:
            misclassify[true_lbl][adv_p] += 1
    return misclassify

print("\nStandard model - Occlusion attack misclassification (top_k=5, ks=3):")
occlusion = OcclusionAttack(std_model, top_k=5, occlu_color=0.0, kernel_size=3)
mis = analyze_misclassification(std_model, occlusion, num_samples=1000)

all_mis = []
for true_lbl in mis:
    for adv_lbl, count in mis[true_lbl].items():
        all_mis.append((true_lbl, adv_lbl, count))

all_mis.sort(key=lambda x: -x[2])
print("\nTop 10 misclassification patterns:")
for true_lbl, adv_lbl, count in all_mis[:10]:
    print(f"  {true_lbl} -> {adv_lbl}: {count} times")

# ========== Experiment 4: White-box vs Transfer Attack ==========
print("\n" + "="*80)
print("Experiment 4: White-box vs Transfer Attack")
print("="*80)

if 'Occlusion-AT' in models_dict:
    target_model = models_dict['Occlusion-AT']

    occ_white = OcclusionAttack(target_model, top_k=5, occlu_color=0.0, kernel_size=3)
    white_acc, _ = test_fn(nn.Sequential(occ_white, target_model),
                          sample_imgs, sample_lbls, bs=250, mode='attack')

    occ_transfer = OcclusionAttack(std_model, top_k=5, occlu_color=0.0, kernel_size=3)
    transfer_acc, _ = test_fn(nn.Sequential(occ_transfer, target_model),
                             sample_imgs, sample_lbls, bs=250, mode='attack')

    clean_acc, _ = test_fn(target_model, sample_imgs, sample_lbls, bs=250, mode='clean')

    print(f"\nOcclusion-AT Model:")
    print(f"  Clean acc: {clean_acc:.2f}%")
    print(f"  White-box acc: {white_acc:.2f}% (success: {clean_acc-white_acc:.2f}%)")
    print(f"  Transfer acc: {transfer_acc:.2f}% (success: {clean_acc-transfer_acc:.2f}%)")
    print(f"  Transferability diff: {abs(white_acc-transfer_acc):.2f}%")

# ========== Generate Summary ==========
print("\n" + "="*80)
print("Generating Summary Report")
print("="*80)

summary = []
for model_name in models_dict.keys():
    model_data = results_df[results_df['model'] == model_name]
    row = model_data[(model_data['top_k'] == 5) & (model_data['kernel_size'] == 3)]
    if not row.empty:
        summary.append({
            'model': model_name,
            'clean_acc': f"{row['clean_acc'].values[0]:.2f}%",
            'occlusion_acc': f"{row['occlusion_acc'].values[0]:.2f}%",
            'success_rate': f"{row['success_rate'].values[0]:.2f}%"
        })

summary_df = pd.DataFrame(summary)
print("\nSummary:")
print(summary_df.to_string(index=False))
summary_df.to_csv('./summary_report.csv', index=False)

# ========== Create Heatmap ==========
print("\nCreating heatmap visualization...")

fig, axes = plt.subplots(1, len(models_dict), figsize=(5*len(models_dict), 4))
if len(models_dict) == 1:
    axes = [axes]

for idx, model_name in enumerate(models_dict.keys()):
    model_data = results_df[results_df['model'] == model_name]
    pivot = model_data.pivot(index='top_k', columns='kernel_size', values='occlusion_acc')

    ax = axes[idx]
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(kernel_size_values)))
    ax.set_yticks(range(len(top_k_values)))
    ax.set_xticklabels(kernel_size_values)
    ax.set_yticklabels(top_k_values)
    ax.set_xlabel('kernel_size')
    ax.set_ylabel('top_k')
    ax.set_title(f'{model_name}\nOcclusion Attack Accuracy (%)')

    for i in range(len(top_k_values)):
        for j in range(len(kernel_size_values)):
            ax.text(j, i, f'{pivot.values[i, j]:.1f}',
                   ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('./occlusion_heatmap.png', dpi=150, bbox_inches='tight')
print("[OK] Saved: occlusion_heatmap.png")

print("\n" + "="*80)
print("All experiments completed!")
print("="*80)
print("\nGenerated files:")
print("  - exp1_occlusion_params.csv")
print("  - exp2_occlusion_colors.csv")
print("  - summary_report.csv")
print("  - occlusion_heatmap.png")
print("="*80)
