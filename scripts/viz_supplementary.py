#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文补充可视化脚本 v2.

按要求拆分：
  1) 最小遮蔽成功案例：每 (model × attack) 单独成图
  2) Saliency 热图 + 遮蔽位置 + 攻击结果：每 model 一张
  3) 四种攻击对比：每 model 一张，仅展示攻击成功样本
  4) 多模型对比：每种 attack 一张，仅展示成功样本，遮蔽位置红色叠加
  5) Adaptive 内部迭代：每种 attack 一张，含最终攻击结果
  6) Fixed vs Adaptive 遮蔽位置对比：每种归因 (Saliency/IG) 一张

图内文字尽量简洁；详细解释（命中率定义、攻击强度数据等）放论文。
所有图保存到 paper_figures/v2/ 与论文 figures/v2/ 子目录。
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
if HAS_CAPTUM:
    from captum.attr import IntegratedGradients

# ============================================================
# 配置
# ============================================================
PAPER_FIG_DIR = os.path.join(ROOT, 'paper_figures', 'v2')
THESIS_FIG_DIR = os.path.join(r'D:\软件\南开大学论文模板2026\figures', 'v2')
MODEL_DIR = os.path.join(ROOT, 'save_model', '50epoch')
DATA_DIR = os.path.join(ROOT, 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 150


# ============================================================
# 模型与攻击注册表
# ============================================================
MODELS = [
    ('Standard',         'mnist_lenet5.pth'),
    ('PGD-AT',           'mnist_lenet5_PGD_0.1_5_AT.pth'),
    ('FGSM-AT',          'mnist_lenet5_FGSM_AT.pth'),
    ('AdaSal-AT',        'mnist_lenet5_AdaptiveSaliencyOcclusionAT_5_3.pth'),
    ('AdaIG-AT',         'mnist_lenet5_AdaptiveIGOcclusionAT_5_3.pth'),
    ('Mix-AT(Sal+PGD)',  'mnist_lenet5_AdaptiveSaliencyPgdMixedAT_0.5_5_3.pth'),
    ('Mix-AT(IG+PGD)',   'mnist_lenet5_AdaptiveMixedAT_0.5_5_3.pth'),
]

ATTACKS = [
    ('FixedSal',  lambda m: SaliencyOcclusionAttack(m, top_k=9, kernel_size=3)),
    ('AdaSal',    lambda m: AdaptiveSaliencyOcclusionAttack(m, N=5, R=3, c=0.0)),
    ('FixedIG',   lambda m: IGFixedAttack(m, top_k=9, kernel_size=3) if HAS_CAPTUM else None),
    ('AdaIG',     lambda m: IGAdaptiveAttack(m, N=5, R=3, c=0.0) if HAS_CAPTUM else None),
]


# ============================================================
# 工具
# ============================================================
def load_model(filename):
    p = os.path.join(MODEL_DIR, filename)
    m = LeNet5()
    ck = torch.load(p, map_location=DEVICE, weights_only=False)
    if isinstance(ck, dict) and 'net' in ck:
        m.load_state_dict(ck['net'])
    else:
        m.load_state_dict(ck)
    return m.to(DEVICE).eval()


def get_pool(n=128, seed=0):
    """取一池样本用于筛选攻击成功案例"""
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


def run_attack_with_mask(model, attack_obj, xs, ys):
    """对一批样本运行攻击，返回 (x_adv, mask, success_arr, mask_count_arr)"""
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


def label_str(true_lbl, pred, conf):
    ok = 'OK' if pred == true_lbl else 'X'
    return f'真:{true_lbl} 预:{pred} {ok}'


def save_fig(fig, subdir, name):
    for d in [os.path.join(PAPER_FIG_DIR, subdir), os.path.join(THESIS_FIG_DIR, subdir)]:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name + '.png'))
    plt.close(fig)


def short(s):
    return s.replace('-', '').replace('+', 'p').replace('(', '').replace(')', '').replace(' ', '')


# ============================================================
# 图1：最小遮蔽成功案例 (每 model × attack 一张)
# ============================================================
def fig_minimal(model, model_name, attack_name, attack_factory, n_cases=4):
    atk = attack_factory(model)
    if atk is None:
        return
    pool_x, pool_y = get_pool(n=128, seed=2)
    x_adv, mask, success, mc = run_attack_with_mask(model, atk, pool_x, pool_y)
    valid = np.where(success)[0]
    if len(valid) == 0:
        # 没有任何成功样本，跳过
        print(f'  [skip] {model_name} × {attack_name} 无成功攻击案例')
        return
    valid = valid[np.argsort(mc[valid])][:n_cases]
    n = len(valid)

    fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.0))
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
        axes[0, c].set_title(f'真:{true_l}', fontsize=10)
        axes[0, c].axis('off')

        axes[1, c].imshow(x_a, cmap='gray', vmin=0, vmax=1)
        axes[1, c].set_title(f'预:{int(pred_a[0])}({conf_a[0]:.0f}%)', fontsize=9, color='red')
        axes[1, c].axis('off')

        axes[2, c].imshow(red_overlay(img, m))
        axes[2, c].set_title(f'{int(m.sum())}px', fontsize=9)
        axes[2, c].axis('off')

    row_labels = ['原图', '攻击后', '遮蔽位置']
    for r, lab in enumerate(row_labels):
        axes[r, 0].text(-0.20, 0.5, lab, transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=10)
    fig.suptitle(f'{model_name} × {attack_name}', fontsize=11)
    plt.tight_layout()
    save_fig(fig, 'min', f'min_{short(model_name)}_{attack_name}')


# ============================================================
# 图2：Saliency 热图 + 遮蔽位置 + 攻击结果（每 model 一张）
# ============================================================
def fig_saliency_overlay(model, model_name, n_samples=5):
    atk = AdaptiveSaliencyOcclusionAttack(model, N=5, R=3, c=0.0)
    pool_x, pool_y = get_pool(n=128, seed=7)
    x_adv, mask, success, _ = run_attack_with_mask(model, atk, pool_x, pool_y)
    # 仅取攻击成功样本
    idxs = np.where(success)[0][:n_samples]
    if len(idxs) == 0:
        idxs = np.arange(min(n_samples, pool_x.shape[0]))
    n = len(idxs)
    xs = pool_x[idxs]
    ys = pool_y[idxs]

    # Saliency
    x_g = xs.detach().requires_grad_(True)
    sal = compute_saliency(model, x_g, ys).detach().cpu().numpy()[:, 0]

    fig, axes = plt.subplots(4, n, figsize=(2.0 * n, 8.2))
    if n == 1:
        axes = axes[:, None]

    hits = []
    for c, idx in enumerate(idxs):
        img = pool_x[idx, 0].cpu().numpy()
        s = sal[c] / (sal[c].max() + 1e-9)
        # top-30 高重要性
        thr = np.sort(s.flatten())[::-1][30]
        high = (s >= thr).astype(float)
        m = mask[idx, 0].cpu().numpy()
        x_a = x_adv[idx, 0].cpu().numpy()
        true_l = int(pool_y[idx])
        pred_a, conf_a = predict(model, x_adv[idx:idx + 1])
        # 命中率：遮蔽中落在高重要性区域的比例
        hit = (m * high).sum() / (m.sum() + 1e-9) * 100
        hits.append(hit)

        axes[0, c].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, c].set_title(f'真:{true_l}', fontsize=9)
        axes[0, c].axis('off')

        axes[1, c].imshow(img, cmap='gray', alpha=0.4)
        axes[1, c].imshow(s, cmap='hot', alpha=0.7)
        axes[1, c].axis('off')

        axes[2, c].imshow(red_overlay(img, m))
        axes[2, c].set_title(f'命中{hit:.0f}%', fontsize=9)
        axes[2, c].axis('off')

        axes[3, c].imshow(x_a, cmap='gray', vmin=0, vmax=1)
        axes[3, c].set_title(f'预:{int(pred_a[0])}', fontsize=9, color='red')
        axes[3, c].axis('off')

    row_labels = ['原图', 'Saliency 热图', '遮蔽位置', '攻击结果']
    for r, lab in enumerate(row_labels):
        axes[r, 0].text(-0.22, 0.5, lab, transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=10)
    fig.suptitle(f'{model_name}：显著性与遮蔽关系  '
                 f'(平均命中率 {np.mean(hits):.1f}%)', fontsize=10.5)
    plt.tight_layout()
    save_fig(fig, 'sal_overlay', f'sal_{short(model_name)}')


# ============================================================
# 图3：四种攻击对比（仅成功案例，每 model 一张）
# ============================================================
def fig_four_attacks(model, model_name, n_cases=5):
    pool_x, pool_y = get_pool(n=128, seed=11)
    runs = []
    for name, factory in ATTACKS:
        atk = factory(model)
        if atk is None:
            runs.append((name, None, None, None))
            continue
        x_adv, mask, success, _ = run_attack_with_mask(model, atk, pool_x, pool_y)
        runs.append((name, x_adv, mask, success))

    # 选取在「至少 2 种攻击下都成功」的样本
    succ_count = np.zeros(pool_x.shape[0], dtype=int)
    for _, _, _, success in runs:
        if success is not None:
            succ_count += success
    cand = np.where(succ_count >= 2)[0]
    if len(cand) == 0:
        cand = np.where(succ_count >= 1)[0]
    if len(cand) == 0:
        print(f'  [skip] {model_name} 无可用成功案例')
        return
    rng = np.random.RandomState(3)
    chosen = rng.choice(cand, min(n_cases, len(cand)), replace=False)

    fig, axes = plt.subplots(len(chosen), 5, figsize=(11, 2.2 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    col_titles = ['原图', 'FixedSal', 'AdaSal', 'FixedIG', 'AdaIG']
    for r, idx in enumerate(chosen):
        true_l = int(pool_y[idx])
        img = pool_x[idx, 0].cpu().numpy()
        axes[r, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[r, 0].set_title(col_titles[0] if r == 0 else '', fontsize=10)
        axes[r, 0].text(-0.2, 0.5, f'真:{true_l}', transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=10)
        axes[r, 0].axis('off')
        for c, (name, x_adv, mask, success) in enumerate(runs):
            ax = axes[r, c + 1]
            if x_adv is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                ax.axis('off')
                continue
            x_a = x_adv[idx, 0].cpu().numpy()
            ax.imshow(x_a, cmap='gray', vmin=0, vmax=1)
            pred_a, _ = predict(model, x_adv[idx:idx + 1])
            color = 'red' if success[idx] else 'green'
            tag = f'预:{int(pred_a[0])}'
            head = col_titles[c + 1] if r == 0 else ''
            ax.set_title(f'{head}\n{tag}', fontsize=9, color=color)
            ax.axis('off')
    fig.suptitle(f'{model_name}：四种遮蔽攻击对比 (绿=失败, 红=成功)', fontsize=10.5)
    plt.tight_layout()
    save_fig(fig, '4atk', f'4atk_{short(model_name)}')


# ============================================================
# 图4：多模型对比（每种 attack 一张，红色叠加遮蔽位置）
# ============================================================
def fig_model_compare(models_dict, attack_name, attack_factory, n_cases=4):
    pool_x, pool_y = get_pool(n=128, seed=13)
    # 用每个模型分别攻击
    results = {}
    for mname, m in models_dict.items():
        atk = attack_factory(m)
        if atk is None:
            results[mname] = None
            continue
        x_adv, mask, success, _ = run_attack_with_mask(m, atk, pool_x, pool_y)
        results[mname] = (x_adv, mask, success)

    # 选取在 Standard 上攻击成功的样本（典型）
    std_res = results.get('Standard')
    if std_res is None:
        print('Standard 缺失，跳过')
        return
    cand = np.where(std_res[2])[0]
    if len(cand) == 0:
        return
    rng = np.random.RandomState(5)
    chosen = rng.choice(cand, min(n_cases, len(cand)), replace=False)

    n_models = len(models_dict)
    fig, axes = plt.subplots(len(chosen), n_models + 1, figsize=(1.9 * (n_models + 1), 2.2 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    for r, idx in enumerate(chosen):
        true_l = int(pool_y[idx])
        img = pool_x[idx, 0].cpu().numpy()
        axes[r, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[r, 0].set_title('原图' if r == 0 else '', fontsize=10)
        axes[r, 0].text(-0.25, 0.5, f'真:{true_l}', transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=10)
        axes[r, 0].axis('off')
        for c, (mname, m) in enumerate(models_dict.items()):
            ax = axes[r, c + 1]
            res = results[mname]
            if res is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center'); ax.axis('off'); continue
            x_adv, mask, success = res
            x_a = x_adv[idx, 0].cpu().numpy()
            m_np = mask[idx, 0].cpu().numpy()
            # 红色叠加：原始灰度 + 红色高亮 mask
            disp = red_overlay(x_a, m_np, alpha=0.6)
            pred_a, _ = predict(m, x_adv[idx:idx + 1])
            color = 'red' if success[idx] else 'green'
            ax.imshow(disp)
            head = mname if r == 0 else ''
            ax.set_title(f'{head}\n预:{int(pred_a[0])}', fontsize=8.5, color=color)
            ax.axis('off')
    fig.suptitle(f'{attack_name} 攻击：多模型遮蔽效果对比 (红块=遮蔽位置)', fontsize=10.5)
    plt.tight_layout()
    save_fig(fig, 'model_cmp', f'cmp_{attack_name}')


# ============================================================
# 图5：Adaptive 内部迭代（每种 attack 一张，含最终结果）
# ============================================================
def _attribution_for_attack(model, x, y, attack_kind):
    """attack_kind: 'sal' or 'ig'，返回展平后的归因排序索引"""
    if attack_kind == 'sal':
        x_g = x.detach().requires_grad_(True)
        attr = compute_saliency(model, x_g, y)
        attr_2d = attr.sum(dim=1)  # [B,H,W]
    else:
        x_ig = x.detach().requires_grad_()
        ig = IntegratedGradients(model)
        a = ig.attribute(x_ig, target=y, n_steps=50, baselines=x_ig * 0).detach()
        attr_2d = a.mean(dim=1)
    return attr_2d


def fig_adaptive_inner(model, model_name, attack_kind, n_samples=3, N=5, R=3):
    """attack_kind: 'sal' or 'ig'，绘制 r=1..R 的累积 mask + 最终攻击结果"""
    if attack_kind == 'ig' and not HAS_CAPTUM:
        return
    pool_x, pool_y = get_pool(n=64, seed=17)
    # 选取攻击成功样本
    if attack_kind == 'sal':
        atk = AdaptiveSaliencyOcclusionAttack(model, N=N, R=R, c=0.0)
        atk_fixed_kind = 'AdaSal'
    else:
        atk = IGAdaptiveAttack(model, N=N, R=R, c=0.0)
        atk_fixed_kind = 'AdaIG'
    x_adv_full, mask_full, success, _ = run_attack_with_mask(model, atk, pool_x, pool_y)
    cand = np.where(success)[0]
    if len(cand) == 0:
        print(f'  [skip] {model_name} {atk_fixed_kind} 无成功案例')
        return
    rng = np.random.RandomState(19)
    chosen = rng.choice(cand, min(n_samples, len(cand)), replace=False)
    xs = pool_x[chosen]
    ys = pool_y[chosen]

    attr_2d = _attribution_for_attack(model, xs, ys, attack_kind)
    bs, H, W = attr_2d.shape
    flat = attr_2d.view(bs, -1)
    _, ranked = torch.sort(flat, dim=1, descending=True)

    fig, axes = plt.subplots(len(chosen), R + 2, figsize=(2.0 * (R + 2), 2.0 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    for s, idx in enumerate(chosen):
        img = pool_x[idx, 0].cpu().numpy()
        true_l = int(pool_y[idx])
        # col 0: 原图
        axes[s, 0].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[s, 0].set_title('原图' if s == 0 else '', fontsize=9)
        axes[s, 0].text(-0.25, 0.5, f'真:{true_l}', transform=axes[s, 0].transAxes,
                        rotation=90, va='center', ha='center', fontsize=9)
        axes[s, 0].axis('off')
        # 各 r 累积叠加
        for r_i, r_val in enumerate(range(1, R + 1)):
            mk = np.zeros((H, W), dtype=float)
            for k in range(N):
                fi = int(ranked[s, k])
                row, col = fi // W, fi % W
                mk[max(0, row - r_val):min(H, row + r_val + 1),
                   max(0, col - r_val):min(W, col + r_val + 1)] += 1
            ax = axes[s, 1 + r_i]
            ax.imshow(img, cmap='gray', alpha=0.4)
            ax.imshow(mk, cmap='hot', alpha=0.75, vmin=0, vmax=N)
            n_px = int((mk > 0).sum())
            head = f'r={r_val}' if s == 0 else ''
            ax.set_title(f'{head}\n{n_px}px', fontsize=8)
            ax.axis('off')
        # 最后一列：最终攻击结果
        x_a = x_adv_full[idx, 0].cpu().numpy()
        pred_a, _ = predict(model, x_adv_full[idx:idx + 1])
        ax = axes[s, -1]
        ax.imshow(x_a, cmap='gray', vmin=0, vmax=1)
        head = '攻击结果' if s == 0 else ''
        ax.set_title(f'{head}\n预:{int(pred_a[0])}', fontsize=9, color='red')
        ax.axis('off')

    title = f'{model_name}：{atk_fixed_kind} 内部迭代 (N={N}, r=1..{R})'
    fig.suptitle(title, fontsize=10.5)
    plt.tight_layout()
    save_fig(fig, 'ada_inner', f'ainner_{atk_fixed_kind}_{short(model_name)}')


# ============================================================
# 图6：Fixed vs Adaptive 遮蔽位置对比（每种归因一张）
# ============================================================
def fig_fix_vs_ada(model, model_name, attribution, n_cases=4):
    """attribution: 'sal' or 'ig'"""
    if attribution == 'sal':
        atk_f = SaliencyOcclusionAttack(model, top_k=9, kernel_size=3)
        atk_a = AdaptiveSaliencyOcclusionAttack(model, N=5, R=3, c=0.0)
        tag = 'Saliency'
    else:
        if not HAS_CAPTUM:
            return
        atk_f = IGFixedAttack(model, top_k=9, kernel_size=3)
        atk_a = IGAdaptiveAttack(model, N=5, R=3, c=0.0)
        tag = 'IG'

    pool_x, pool_y = get_pool(n=64, seed=23)
    x_f, m_f, suc_f, _ = run_attack_with_mask(model, atk_f, pool_x, pool_y)
    x_a, m_a, suc_a, _ = run_attack_with_mask(model, atk_a, pool_x, pool_y)
    # 选两种都成功的
    both = np.where(suc_f & suc_a)[0]
    if len(both) == 0:
        both = np.where(suc_f | suc_a)[0]
    if len(both) == 0:
        return
    rng = np.random.RandomState(29)
    chosen = rng.choice(both, min(n_cases, len(both)), replace=False)

    fig, axes = plt.subplots(len(chosen), 5, figsize=(11, 2.2 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    for r, idx in enumerate(chosen):
        true_l = int(pool_y[idx])
        img = pool_x[idx, 0].cpu().numpy()
        mf = m_f[idx, 0].cpu().numpy()
        ma = m_a[idx, 0].cpu().numpy()
        overlap = ((mf > 0) & (ma > 0)).astype(float)
        only_f = ((mf > 0) & (ma == 0)).astype(float)
        only_a = ((mf == 0) & (ma > 0)).astype(float)
        pred_f, _ = predict(model, x_f[idx:idx + 1])
        pred_a, _ = predict(model, x_a[idx:idx + 1])

        # 原图
        ax = axes[r, 0]; ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title('原图' if r == 0 else '', fontsize=10)
        ax.text(-0.22, 0.5, f'真:{true_l}', transform=ax.transAxes,
                rotation=90, va='center', ha='center', fontsize=10)
        ax.axis('off')
        # Fixed 攻击结果
        ax = axes[r, 1]; ax.imshow(x_f[idx, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        col = 'red' if suc_f[idx] else 'green'
        ax.set_title((f'Fixed\n预:{int(pred_f[0])}' if r == 0 else f'预:{int(pred_f[0])}'),
                     fontsize=9, color=col)
        ax.axis('off')
        # Adaptive 攻击结果
        ax = axes[r, 2]; ax.imshow(x_a[idx, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        col = 'red' if suc_a[idx] else 'green'
        ax.set_title((f'Adaptive\n预:{int(pred_a[0])}' if r == 0 else f'预:{int(pred_a[0])}'),
                     fontsize=9, color=col)
        ax.axis('off')
        # 遮蔽位置叠加
        rgb = np.stack([img, img, img], axis=-1) * 0.4
        rgb[..., 0] += only_f * 0.9
        rgb[..., 2] += only_a * 0.9
        rgb[..., 1] += overlap * 0.9
        rgb = np.clip(rgb, 0, 1)
        ax = axes[r, 3]; ax.imshow(rgb)
        n_ov = int(overlap.sum())
        head = '红=Fixed 蓝=Ada\n绿=重叠' if r == 0 else ''
        ax.set_title(f'{head}\n重叠:{n_ov}px', fontsize=8.5)
        ax.axis('off')
        # 各自像素数
        ax = axes[r, 4]
        ax.bar(['Fixed', 'Ada'], [int(mf.sum()), int(ma.sum())],
               color=['#E74C3C', '#3498DB'])
        ax.set_ylim(0, max(mf.sum(), ma.sum()) * 1.4 + 5)
        ax.set_title('遮蔽像素数' if r == 0 else '', fontsize=9)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    fig.suptitle(f'{model_name}：{tag} 归因下 Fixed vs Adaptive', fontsize=10.5)
    plt.tight_layout()
    save_fig(fig, 'fix_vs_ada', f'fva_{tag}_{short(model_name)}')


# ============================================================
# 主流程
# ============================================================
def main():
    print(f'Device: {DEVICE} | captum: {HAS_CAPTUM}')
    # 加载所有模型
    models = {}
    for name, fn in MODELS:
        try:
            models[name] = load_model(fn)
            print(f'已加载 {name} <- {fn}')
        except Exception as e:
            print(f'加载 {name} 失败: {e}')

    # ---------- 图1: 最小遮蔽 ----------
    print('\n[1/6] 最小遮蔽案例...')
    for mname, m in models.items():
        for aname, fac in ATTACKS:
            try:
                fig_minimal(m, mname, aname, fac)
            except Exception as e:
                print(f'  失败 {mname}×{aname}: {e}')

    # ---------- 图2: Saliency overlay ----------
    print('\n[2/6] 显著性叠加...')
    for mname, m in models.items():
        try:
            fig_saliency_overlay(m, mname)
        except Exception as e:
            print(f'  失败 {mname}: {e}')

    # ---------- 图3: 四攻击对比 ----------
    print('\n[3/6] 四种攻击对比...')
    for mname, m in models.items():
        try:
            fig_four_attacks(m, mname)
        except Exception as e:
            print(f'  失败 {mname}: {e}')

    # ---------- 图4: 多模型对比 ----------
    print('\n[4/6] 多模型对比...')
    for aname, fac in ATTACKS:
        try:
            fig_model_compare(models, aname, fac)
        except Exception as e:
            print(f'  失败 {aname}: {e}')

    # ---------- 图5: Adaptive 内部 ----------
    print('\n[5/6] Adaptive 内部迭代...')
    for mname, m in models.items():
        for kind in ['sal', 'ig']:
            try:
                fig_adaptive_inner(m, mname, kind)
            except Exception as e:
                print(f'  失败 {mname}/{kind}: {e}')

    # ---------- 图6: Fixed vs Adaptive ----------
    print('\n[6/6] Fixed vs Adaptive...')
    for mname, m in models.items():
        for attr in ['sal', 'ig']:
            try:
                fig_fix_vs_ada(m, mname, attr)
            except Exception as e:
                print(f'  失败 {mname}/{attr}: {e}')

    print(f'\n全部图已保存到 {PAPER_FIG_DIR} 与 {THESIS_FIG_DIR}')


if __name__ == '__main__':
    main()
