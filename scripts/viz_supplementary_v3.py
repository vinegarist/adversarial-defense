#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文补充可视化脚本 v3 (重绘版).

本版本相对 v2 的关键变化（按导师/作者要求）：
  1. 统一命名：
     - 攻击：Fixed-Saliency / Adaptive-Saliency / Fixed-IG / Adaptive-IG
     - 模型：Standard / PGD-AT / FGSM-AT / Adaptive-Saliency-AT / Adaptive-IG-AT /
              Mix-AT(Sal+PGD) / Mix-AT(IG+PGD)
  2. 参数对齐：Fixed top_k = Adaptive N = 5；Fixed kernel_size = 2R + 1 = 7；R=3。
  3. 所有图均显示：真实标签 + 预测 + 置信度 + OK/X 标记。
  4. 涉及 mask 的图均显示红块色例并在标题或图例中注明攻击参数。
  5. 每张图最多 5 个样本；图幅控制在 textwidth 95% 以内。
  6. 输出文件名沿用旧的短码以保持论文 \\includegraphics 引用不变：
       FixedSal / AdaSal / FixedIG / AdaIG
       PGDAT / FGSMAT / AdaSalAT / AdaIGAT / MixATSalpPGD / MixATIGpPGD
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import LeNet5
from occlusion_attack import (
    compute_saliency,
    SaliencyOcclusionAttack,
    AdaptiveSaliencyOcclusionAttack,
    OcclusionAttack as IGFixedAttack,
    AdaptiveOcclusionAttack as IGAdaptiveAttack,
    HAS_CAPTUM,
)

# ============================================================
# 配置
# ============================================================
PAPER_FIG_DIR = os.path.join(ROOT, 'paper_figures', 'v2')
THESIS_FIG_DIR = os.path.join(r'D:\软件\南开大学论文模板2026\figures', 'v2')
MODEL_DIR = os.path.join(ROOT, 'save_model', '50epoch')
DATA_DIR = os.path.join(ROOT, 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 统一对齐的攻击参数 ──
N_FIXED = 5      # Fixed-Saliency / Fixed-IG 的 top_k
N_ADA = 5        # Adaptive-Saliency / Adaptive-IG 的 N
R_ADA = 3        # Adaptive 的最大半径
KERNEL = 2 * R_ADA + 1  # 7
PARAM_TXT = f'k=N={N_FIXED}, R={R_ADA}, kernel={KERNEL}'

N_SAMPLES = 5    # 每图统一 5 个样本

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 150
# 增大论文图片字体
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14


# ============================================================
# 模型与攻击注册表（统一名称）
# ============================================================
MODELS = [
    ('Standard',                'Standard',             'mnist_lenet5.pth'),
    ('PGD-AT',                  'PGDAT',                'mnist_lenet5_PGD_0.1_5_AT.pth'),
    ('FGSM-AT',                 'FGSMAT',               'mnist_lenet5_FGSM_AT.pth'),
    ('Adaptive-Saliency-AT',    'AdaSalAT',             'mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth'),
    ('Adaptive-IG-AT',          'AdaIGAT',              'mnist_lenet5_AdaptiveIGOcclusionAT_5_3.pth'),
    ('Mix-AT(Sal+PGD)',         'MixATSalpPGD',         'mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth'),
    ('Mix-AT(IG+PGD)',          'MixATIGpPGD',          'mnist_lenet5_AdaptiveMixedAT_0.5_5_3.pth'),
]


def make_attacks(model):
    """返回 4 种攻击：(显示名, 短码, 实例)"""
    out = [
        ('Fixed-Saliency',    'FixedSal',
         SaliencyOcclusionAttack(model, top_k=N_FIXED, kernel_size=KERNEL)),
        ('Adaptive-Saliency', 'AdaSal',
         AdaptiveSaliencyOcclusionAttack(model, N=N_ADA, R=R_ADA, c=0.0)),
    ]
    if HAS_CAPTUM:
        out.append(('Fixed-IG', 'FixedIG',
                    IGFixedAttack(model, top_k=N_FIXED, kernel_size=KERNEL)))
        out.append(('Adaptive-IG', 'AdaIG',
                    IGAdaptiveAttack(model, N=N_ADA, R=R_ADA, c=0.0)))
    else:
        out.append(('Fixed-IG', 'FixedIG', None))
        out.append(('Adaptive-IG', 'AdaIG', None))
    return out


# ============================================================
# 工具
# ============================================================
def load_model(filename):
    p = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(p):
        return None
    m = LeNet5()
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    if isinstance(ck, dict) and 'net' in ck:
        m.load_state_dict(ck['net'])
    else:
        m.load_state_dict(ck)
    return m.to(DEVICE).eval()


def get_pool(n=128, seed=0):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(testset), n, replace=False)
    xs, ys = [], []
    for i in idx:
        x, y = testset[int(i)]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(DEVICE), torch.tensor(ys, dtype=torch.long, device=DEVICE)


def predict(model, x):
    with torch.no_grad():
        out = model(x)
        prob = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)
        conf = prob.gather(1, pred.unsqueeze(1)).squeeze(1)
    return pred.cpu().numpy(), (conf * 100).cpu().numpy()


def run_attack(model, attack_obj, xs, ys):
    """运行攻击；返回 (x_adv, mask, success_arr, mask_count_arr)"""
    if attack_obj is None:
        return None, None, None, None
    x_adv = attack_obj((xs, ys))
    mask = ((x_adv - xs).abs() > 1e-6).float()
    with torch.no_grad():
        pred = model(x_adv).argmax(dim=1)
    success = (pred != ys).cpu().numpy()
    mc = mask.view(xs.shape[0], -1).sum(dim=1).cpu().numpy()
    return x_adv, mask, success, mc


def red_overlay(img, mask, alpha=0.7):
    """灰度原图 + 红色 mask 叠加"""
    rgb = np.stack([img, img, img], axis=-1).astype(float)
    m = mask.astype(float)
    rgb[..., 0] = np.clip(rgb[..., 0] + m * alpha, 0, 1)
    rgb[..., 1] *= (1 - m * 0.5)
    rgb[..., 2] *= (1 - m * 0.5)
    return rgb


def add_mask_legend(fig, color='red', label='遮蔽位置', loc='upper right'):
    """为整张图加 mask 颜色图例"""
    patch = mpatches.Patch(color=color, label=label)
    fig.legend(handles=[patch], loc=loc, fontsize=16, framealpha=0.85)


def status_color(pred, true_l):
    return 'red' if pred != true_l else 'green'


def title_pred(pred, conf, true_l):
    ok = 'OK' if pred == true_l else 'X'
    return f'真:{true_l} 预:{pred}({conf:.0f}%) {ok}'


def title_pred_only(pred, conf):
    return f'{pred}({conf:.0f}%)'


def title_pred_compact(pred, conf, true_l):
    return f'{true_l}->{pred}({conf:.0f}%)'


def model_header(name):
    return {
        'Standard': 'Standard',
        'PGD-AT': 'PGD-AT',
        'FGSM-AT': 'FGSM-AT',
        'Adaptive-Saliency-AT': 'Ada-Sal\nAT',
        'Adaptive-IG-AT': 'Ada-IG\nAT',
        'Mix-AT(Sal+PGD)': 'Mix-AT\n(Sal+PGD)',
        'Mix-AT(IG+PGD)': 'Mix-AT\n(IG+PGD)',
    }.get(name, name)


def attribution_header(prefix, attribution):
    short_attr = 'Sal' if attribution == 'Saliency' else 'IG'
    return f'{prefix}-{short_attr}'


def save_fig(fig, subdir, name):
    for d in [os.path.join(PAPER_FIG_DIR, subdir), os.path.join(THESIS_FIG_DIR, subdir)]:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name + '.png'))
    plt.close(fig)


# ============================================================
# 图1：最小遮蔽成功案例 (每 model × attack 一张)
# ============================================================
def fig_minimal(model, model_name, model_short, attack_name, attack_short, attack_obj, n_cases=N_SAMPLES):
    if attack_obj is None:
        return
    pool_x, pool_y = get_pool(n=128, seed=2)
    x_adv, mask, success, mc = run_attack(model, attack_obj, pool_x, pool_y)
    valid = np.where(success)[0]
    if len(valid) == 0:
        # 模型对该攻击完全鲁棒 — 改用预测变化最大的样本（即使仍预测正确）以提供对比
        with torch.no_grad():
            p_orig = model(pool_x).softmax(dim=1)
            p_adv = model(x_adv).softmax(dim=1)
            kl = (p_orig * (p_orig.clamp_min(1e-9).log() - p_adv.clamp_min(1e-9).log())).sum(dim=1)
        valid = kl.argsort(descending=True).cpu().numpy()[:n_cases]
        no_success = True
    else:
        valid = valid[np.argsort(mc[valid])][:n_cases]
        no_success = False
    n = len(valid)

    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7.0))
    if n == 1:
        axes = axes[:, None]

    for c, idx in enumerate(valid):
        true_l = int(pool_y[idx])
        pred_o, conf_o = predict(model, pool_x[idx:idx + 1])
        pred_a, conf_a = predict(model, x_adv[idx:idx + 1])
        img = pool_x[idx, 0].cpu().numpy()
        m = mask[idx, 0].cpu().numpy()
        x_a = x_adv[idx, 0].cpu().numpy()

        axes[0, c].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, c].set_title(title_pred(int(pred_o[0]), float(conf_o[0]), true_l),
                             fontsize=15, color=status_color(int(pred_o[0]), true_l))
        axes[0, c].axis('off')

        axes[1, c].imshow(x_a, cmap='gray', vmin=0, vmax=1)
        axes[1, c].set_title(title_pred(int(pred_a[0]), float(conf_a[0]), true_l),
                             fontsize=15, color=status_color(int(pred_a[0]), true_l))
        axes[1, c].axis('off')

        axes[2, c].imshow(red_overlay(img, m))
        axes[2, c].set_title(f'{int(m.sum())}px', fontsize=15)
        axes[2, c].axis('off')

    for r, lab in enumerate(['原图', '攻击后', '遮蔽位置']):
        axes[r, 0].text(-0.25, 0.5, lab, transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=17)

    suc_rate = success.mean() * 100 if success is not None else 0.0
    suffix = '（无成功攻击：展示扰动最大样本）' if no_success else ''
    plt.tight_layout()
    save_fig(fig, 'min', f'min_{model_short}_{attack_short}')


# ============================================================
# 图2：归因热图 + 遮蔽位置 + 攻击结果（每 model 一张，针对 Adaptive-Saliency）
# ============================================================
def fig_saliency_overlay(model, model_name, model_short, n=N_SAMPLES):
    atk = AdaptiveSaliencyOcclusionAttack(model, N=N_ADA, R=R_ADA, c=0.0)
    pool_x, pool_y = get_pool(n=128, seed=7)
    x_adv, mask, success, _ = run_attack(model, atk, pool_x, pool_y)
    idxs = np.where(success)[0][:n]
    if len(idxs) == 0:
        idxs = np.arange(min(n, pool_x.shape[0]))
    n = len(idxs)
    xs = pool_x[idxs]
    ys = pool_y[idxs]

    x_g = xs.detach().requires_grad_(True)
    sal = compute_saliency(model, x_g, ys).detach().cpu().numpy()[:, 0]

    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 9.2))
    if n == 1:
        axes = axes[:, None]

    hits = []
    for c, idx in enumerate(idxs):
        img = pool_x[idx, 0].cpu().numpy()
        s = sal[c] / (sal[c].max() + 1e-9)
        thr = np.sort(s.flatten())[::-1][30]
        high = (s >= thr).astype(float)
        m = mask[idx, 0].cpu().numpy()
        x_a = x_adv[idx, 0].cpu().numpy()
        true_l = int(pool_y[idx])
        pred_o, conf_o = predict(model, pool_x[idx:idx + 1])
        pred_a, conf_a = predict(model, x_adv[idx:idx + 1])
        hit = (m * high).sum() / (m.sum() + 1e-9) * 100
        hits.append(hit)

        axes[0, c].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, c].set_title(title_pred(int(pred_o[0]), float(conf_o[0]), true_l),
                             fontsize=15, color=status_color(int(pred_o[0]), true_l))
        axes[0, c].axis('off')

        axes[1, c].imshow(img, cmap='gray', alpha=0.4)
        axes[1, c].imshow(s, cmap='hot', alpha=0.7)
        axes[1, c].axis('off')

        axes[2, c].imshow(red_overlay(img, m))
        axes[2, c].set_title(f'命中{hit:.0f}% / {int(m.sum())}px', fontsize=15)
        axes[2, c].axis('off')

        axes[3, c].imshow(x_a, cmap='gray', vmin=0, vmax=1)
        axes[3, c].set_title(title_pred(int(pred_a[0]), float(conf_a[0]), true_l),
                             fontsize=15, color=status_color(int(pred_a[0]), true_l))
        axes[3, c].axis('off')

    for r, lab in enumerate(['原图', '显著性热图', '遮蔽位置', '攻击结果']):
        axes[r, 0].text(-0.25, 0.5, lab, transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=17)

    plt.tight_layout()
    save_fig(fig, 'sal_overlay', f'sal_{model_short}')


# ============================================================
# 图3：四种攻击对比（每 model 一张）
# ============================================================
def fig_four_attacks(model, model_name, model_short, n_cases=N_SAMPLES):
    pool_x, pool_y = get_pool(n=128, seed=11)
    runs = []
    for aname, ashort, atk in make_attacks(model):
        runs.append((aname, ashort) + run_attack(model, atk, pool_x, pool_y))
    # 选择至少 2 个攻击成功的样本
    succ = np.zeros(pool_x.shape[0], dtype=int)
    for _, _, _, _, success, _ in runs:
        if success is not None:
            succ += success
    cand = np.where(succ >= 2)[0]
    if len(cand) == 0:
        cand = np.where(succ >= 1)[0]
    if len(cand) == 0:
        cand = np.arange(pool_x.shape[0])  # 全部模型都防御成功，随机展示
    rng = np.random.RandomState(3)
    chosen = rng.choice(cand, min(n_cases, len(cand)), replace=False)

    fig, axes = plt.subplots(len(chosen), 5, figsize=(11.5, 2.3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    headers = ['原图'] + [r[0] for r in runs]

    overall_success = {r[0]: (r[4].mean() * 100 if r[4] is not None else None) for r in runs}

    for r_idx, idx in enumerate(chosen):
        true_l = int(pool_y[idx])
        img = pool_x[idx, 0].cpu().numpy()
        pred_o, conf_o = predict(model, pool_x[idx:idx + 1])
        axes[r_idx, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[r_idx, 0].set_title(
            (headers[0] + '\n' if r_idx == 0 else '')
            + title_pred_compact(int(pred_o[0]), float(conf_o[0]), true_l),
            fontsize=20, color=status_color(int(pred_o[0]), true_l))
        axes[r_idx, 0].axis('off')

        for c, (aname, ashort, x_adv, mask, success, mc) in enumerate(runs):
            ax = axes[r_idx, c + 1]
            if x_adv is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.axis('off')
                continue
            x_a = x_adv[idx, 0].cpu().numpy()
            ax.imshow(x_a, cmap='gray', vmin=0, vmax=1)
            pred_a, conf_a = predict(model, x_adv[idx:idx + 1])
            head = headers[c + 1] + '\n' if r_idx == 0 else ''
            ax.set_title(
                head + title_pred_compact(int(pred_a[0]), float(conf_a[0]), true_l),
                fontsize=20, color=status_color(int(pred_a[0]), true_l))
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.18)
    save_fig(fig, '4atk', f'4atk_{model_short}')


# ============================================================
# 图4：多模型对比（每种 attack 一张）
# ============================================================
def fig_model_compare(models_dict, attack_name, attack_short, attack_factory, n_cases=N_SAMPLES):
    pool_x, pool_y = get_pool(n=128, seed=13)
    results = {}
    for mname, mshort, m in models_dict:
        atk = attack_factory(m)
        if atk is None:
            results[(mname, mshort)] = None
            continue
        results[(mname, mshort)] = run_attack(m, atk, pool_x, pool_y)

    std_key = ('Standard', 'Standard')
    if std_key not in results or results[std_key] is None:
        return
    std_res = results[std_key]
    cand = np.where(std_res[2])[0]
    if len(cand) == 0:
        return
    rng = np.random.RandomState(5)
    chosen = rng.choice(cand, min(n_cases, len(cand)), replace=False)

    n_models = len(models_dict)
    fig, axes = plt.subplots(len(chosen), n_models + 1, figsize=(2.25 * (n_models + 1), 2.75 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    headers = ['原图'] + [model_header(m[0]) for m in models_dict]

    for r_idx, idx in enumerate(chosen):
        true_l = int(pool_y[idx])
        img = pool_x[idx, 0].cpu().numpy()
        axes[r_idx, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[r_idx, 0].set_title(headers[0] if r_idx == 0 else '', fontsize=26)
        axes[r_idx, 0].text(-0.25, 0.5, f'真:{true_l}', transform=axes[r_idx, 0].transAxes,
                            rotation=90, va='center', ha='center', fontsize=26)
        axes[r_idx, 0].axis('off')

        for c, (mname, mshort, m) in enumerate(models_dict):
            ax = axes[r_idx, c + 1]
            res = results[(mname, mshort)]
            if res is None or res[0] is None:
                ax.axis('off')
                continue
            x_adv, mask, success, _ = res
            mk = mask[idx, 0].cpu().numpy()
            # 在该模型自己的 mask 上叠加红色，显示该模型的攻击结果
            img_with_mask = red_overlay(img, mk)
            ax.imshow(img_with_mask)
            pred_a, conf_a = predict(m, x_adv[idx:idx + 1])
            head = headers[c + 1] + '\n' if r_idx == 0 else ''
            color = 'red' if success[idx] else 'green'
            ax.set_title(
                head + title_pred_only(int(pred_a[0]), float(conf_a[0])),
                fontsize=26, color=color)
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.36)
    save_fig(fig, 'model_cmp', f'cmp_{attack_short}')


# ============================================================
# 图5：自适应攻击的内部演化（每 attack × model 一张）
# ============================================================
def fig_adaptive_inner(model, model_name, model_short, attack_kind, n_samples=N_SAMPLES):
    """attack_kind: 'AdaSal' 或 'AdaIG'"""
    if attack_kind == 'AdaSal':
        atk_full = AdaptiveSaliencyOcclusionAttack(model, N=N_ADA, R=R_ADA, c=0.0)
        atk_label = 'Adaptive-Saliency'
    else:
        if not HAS_CAPTUM:
            return
        atk_full = IGAdaptiveAttack(model, N=N_ADA, R=R_ADA, c=0.0)
        atk_label = 'Adaptive-IG'

    pool_x, pool_y = get_pool(n=128, seed=17)
    x_adv_full, _, succ_full, _ = run_attack(model, atk_full, pool_x, pool_y)
    cand = np.where(succ_full)[0]
    if len(cand) == 0:
        cand = np.arange(pool_x.shape[0])
    rng = np.random.RandomState(8)
    chosen = rng.choice(cand, min(n_samples, len(cand)), replace=False)

    # 对每个 n (1..N) 重新生成攻击；R 固定为 R_ADA
    snapshots = {}  # n -> (x_adv, mask)
    for n_iter in range(1, N_ADA + 1):
        if attack_kind == 'AdaSal':
            atk_n = AdaptiveSaliencyOcclusionAttack(model, N=n_iter, R=R_ADA, c=0.0)
        else:
            atk_n = IGAdaptiveAttack(model, N=n_iter, R=R_ADA, c=0.0)
        x_adv_n, mask_n, _, _ = run_attack(model, atk_n, pool_x, pool_y)
        snapshots[n_iter] = (x_adv_n, mask_n)

    n_cols = N_ADA + 1  # 原图 + N 步
    fig, axes = plt.subplots(len(chosen), n_cols, figsize=(1.6 * n_cols, 2.2 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    for r_idx, idx in enumerate(chosen):
        true_l = int(pool_y[idx])
        img = pool_x[idx, 0].cpu().numpy()
        axes[r_idx, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[r_idx, 0].set_title('原图' if r_idx == 0 else '', fontsize=22)
        axes[r_idx, 0].text(-0.25, 0.5, f'真:{true_l}', transform=axes[r_idx, 0].transAxes,
                            rotation=90, va='center', ha='center', fontsize=22)
        axes[r_idx, 0].axis('off')

        for c in range(N_ADA):
            n_iter = c + 1
            x_adv_n, mask_n = snapshots[n_iter]
            x_a = x_adv_n[idx, 0].cpu().numpy()
            mk = mask_n[idx, 0].cpu().numpy()
            ax = axes[r_idx, c + 1]
            ax.imshow(red_overlay(x_a, mk))
            pred_a, conf_a = predict(model, x_adv_n[idx:idx + 1])
            head = f'n={n_iter}\n' if r_idx == 0 else ''
            ax.set_title(head + title_pred_only(int(pred_a[0]), float(conf_a[0])),
                         fontsize=22, color=status_color(int(pred_a[0]), true_l))
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.18)
    save_fig(fig, 'ada_inner', f'ainner_{attack_kind}_{model_short}')


# ============================================================
# 图6：Fixed vs Adaptive 遮蔽位置对比（每 attribution × model 一张）
# ============================================================
def fig_fix_vs_ada(model, model_name, model_short, attribution, n_samples=N_SAMPLES):
    """attribution: 'Saliency' 或 'IG'"""
    if attribution == 'Saliency':
        fixed_atk = SaliencyOcclusionAttack(model, top_k=N_FIXED, kernel_size=KERNEL)
        ada_atk = AdaptiveSaliencyOcclusionAttack(model, N=N_ADA, R=R_ADA, c=0.0)
    else:
        if not HAS_CAPTUM:
            return
        fixed_atk = IGFixedAttack(model, top_k=N_FIXED, kernel_size=KERNEL)
        ada_atk = IGAdaptiveAttack(model, N=N_ADA, R=R_ADA, c=0.0)

    pool_x, pool_y = get_pool(n=128, seed=19)
    x_f, m_f, suc_f, mc_f = run_attack(model, fixed_atk, pool_x, pool_y)
    x_a, m_a, suc_a, mc_a = run_attack(model, ada_atk, pool_x, pool_y)

    # 选取在两种攻击下都被攻击的样本
    cand = np.where(suc_f | suc_a)[0]
    if len(cand) == 0:
        cand = np.arange(pool_x.shape[0])
    rng = np.random.RandomState(11)
    chosen = rng.choice(cand, min(n_samples, len(cand)), replace=False)

    fig, axes = plt.subplots(len(chosen), 5, figsize=(12.0, 2.95 * len(chosen)),
                             gridspec_kw={'width_ratios': [0.42, 1, 1, 1, 1]})
    if len(chosen) == 1:
        axes = axes[None, :]

    headers = [
        '',
        f'原图',
        f'{attribution_header("Fixed", attribution)}\n(k={N_FIXED},\nkernel={KERNEL})',
        f'{attribution_header("Ada", attribution)}\n(N={N_ADA},\nR={R_ADA})',
        f'重叠区域'
    ]

    for r_idx, idx in enumerate(chosen):
        true_l = int(pool_y[idx])
        img = pool_x[idx, 0].cpu().numpy()
        mf = m_f[idx, 0].cpu().numpy()
        ma = m_a[idx, 0].cpu().numpy()

        axes[r_idx, 0].axis('off')
        axes[r_idx, 0].text(0.5, 0.5, f'{true_l}', transform=axes[r_idx, 0].transAxes,
                            rotation=90, va='center', ha='center', fontsize=25)

        axes[r_idx, 1].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[r_idx, 1].set_title(headers[1] if r_idx == 0 else '', fontsize=24)
        axes[r_idx, 1].axis('off')

        pred_f, conf_f = predict(model, x_f[idx:idx + 1])
        axes[r_idx, 2].imshow(red_overlay(img, mf))
        head = headers[2] + '\n' if r_idx == 0 else ''
        axes[r_idx, 2].set_title(head + f'{int(mf.sum())}px ' +
                                 title_pred_only(int(pred_f[0]), float(conf_f[0])),
                                 fontsize=25, color=status_color(int(pred_f[0]), true_l))
        axes[r_idx, 2].axis('off')

        pred_a, conf_a = predict(model, x_a[idx:idx + 1])
        rgb = np.stack([img, img, img], axis=-1).astype(float)
        rgb[..., 2] = np.clip(rgb[..., 2] + ma * 0.7, 0, 1)
        rgb[..., 0] *= (1 - ma * 0.5)
        rgb[..., 1] *= (1 - ma * 0.5)
        axes[r_idx, 3].imshow(rgb)
        head = headers[3] + '\n' if r_idx == 0 else ''
        axes[r_idx, 3].set_title(head + f'{int(ma.sum())}px ' +
                                 title_pred_only(int(pred_a[0]), float(conf_a[0])),
                                 fontsize=25, color=status_color(int(pred_a[0]), true_l))
        axes[r_idx, 3].axis('off')

        rgb2 = np.stack([img, img, img], axis=-1).astype(float)
        rgb2[..., 0] = np.clip(rgb2[..., 0] + mf * 0.7, 0, 1)
        rgb2[..., 2] = np.clip(rgb2[..., 2] + ma * 0.7, 0, 1)
        rgb2[..., 1] *= (1 - (mf + ma).clip(0, 1) * 0.5)
        overlap = (mf > 0) & (ma > 0)
        ovr_pct = overlap.sum() / max(1, ((mf > 0) | (ma > 0)).sum()) * 100
        axes[r_idx, 4].imshow(rgb2)
        head = headers[4] + '\n' if r_idx == 0 else ''
        axes[r_idx, 4].set_title(head + f'IoU={ovr_pct:.0f}%', fontsize=23)
        axes[r_idx, 4].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.36)
    save_fig(fig, 'fix_vs_ada', f'fva_{attribution}_{model_short}')


# ============================================================
# 主流程
# ============================================================
def main():
    print(f'>>> Visualization v3 — params: {PARAM_TXT}, samples per fig = {N_SAMPLES}')

    # 加载所有模型（缺失模型会被跳过）
    models = []
    for mname, mshort, fn in MODELS:
        m = load_model(fn)
        if m is None:
            print(f'  [skip] 模型缺失: {fn}')
            continue
        models.append((mname, mshort, m))
    print(f'  已加载 {len(models)} 个模型')

    # 1. 最小遮蔽 (model × 4 attacks)
    print('\n--- [1/6] 最小遮蔽成功案例 ---')
    for mname, mshort, m in models:
        atks = make_attacks(m)
        for aname, ashort, atk in atks:
            print(f'  min: {mname} × {aname}')
            fig_minimal(m, mname, mshort, aname, ashort, atk)

    # 2. 显著性热图 + 遮蔽 (每 model 一张)
    print('\n--- [2/6] 显著性热图叠加 ---')
    for mname, mshort, m in models:
        print(f'  sal: {mname}')
        fig_saliency_overlay(m, mname, mshort)

    # 3. 四种攻击对比
    print('\n--- [3/6] 四种攻击对比 ---')
    for mname, mshort, m in models:
        print(f'  4atk: {mname}')
        fig_four_attacks(m, mname, mshort)

    # 4. 多模型对比 (每种 attack 一张)
    print('\n--- [4/6] 多模型对比 ---')
    factories = {
        'FixedSal': lambda m: SaliencyOcclusionAttack(m, top_k=N_FIXED, kernel_size=KERNEL),
        'AdaSal':   lambda m: AdaptiveSaliencyOcclusionAttack(m, N=N_ADA, R=R_ADA, c=0.0),
    }
    if HAS_CAPTUM:
        factories['FixedIG'] = lambda m: IGFixedAttack(m, top_k=N_FIXED, kernel_size=KERNEL)
        factories['AdaIG']   = lambda m: IGAdaptiveAttack(m, N=N_ADA, R=R_ADA, c=0.0)
    name_map = {'FixedSal': 'Fixed-Saliency', 'AdaSal': 'Adaptive-Saliency',
                'FixedIG': 'Fixed-IG', 'AdaIG': 'Adaptive-IG'}
    for ashort, factory in factories.items():
        print(f'  cmp: {ashort}')
        fig_model_compare(models, name_map[ashort], ashort, factory)

    # 5. Adaptive 内部演化
    print('\n--- [5/6] Adaptive 内部演化 ---')
    for mname, mshort, m in models:
        print(f'  ainner AdaSal: {mname}')
        fig_adaptive_inner(m, mname, mshort, 'AdaSal')
        if HAS_CAPTUM:
            print(f'  ainner AdaIG: {mname}')
            fig_adaptive_inner(m, mname, mshort, 'AdaIG')

    # 6. Fix vs Ada
    print('\n--- [6/6] Fixed vs Adaptive 遮蔽位置 ---')
    for mname, mshort, m in models:
        print(f'  fva Sal: {mname}')
        fig_fix_vs_ada(m, mname, mshort, 'Saliency')
        if HAS_CAPTUM:
            print(f'  fva IG: {mname}')
            fig_fix_vs_ada(m, mname, mshort, 'IG')

    print('\n>>> 全部完成')


if __name__ == '__main__':
    main()
