#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""计算 Mix-AT(Adaptive-IG+PGD) 的迁移攻击准确率."""

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models import LeNet5
from adversarial_training import LinfPGD
from loss import CWLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def load_model(path):
    """加载模型."""
    model = LeNet5()
    ckpt = torch.load(path, map_location=DEVICE)
    if 'net' in ckpt:
        model.load_state_dict(ckpt['net'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    model.eval()
    return model

def test_accuracy(model, x, y):
    """计算准确率."""
    with torch.no_grad():
        pred = model(x).argmax(dim=1)
        acc = (pred == y).float().mean().item() * 100
    return acc

def get_test_data(batch_size=250):
    """获取测试数据."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])
    testset = datasets.MNIST(os.path.join(ROOT, 'data'), train=False, download=False, transform=transform)

    imgs = []
    lbls = []
    for i in range(len(testset)):
        x, y = testset[i]
        imgs.append(x)
        lbls.append(y)
    imgs = torch.stack(imgs).to(DEVICE)
    lbls = torch.tensor(lbls, dtype=torch.long).to(DEVICE)
    return imgs, lbls

def main():
    print('>>> 计算 Mix-AT(Adaptive-IG+PGD) 迁移攻击准确率')

    # 加载模型
    std_model_path = os.path.join(ROOT, 'save_model/50epoch/mnist_lenet5.pth')
    mix_ig_pgd_path = os.path.join(ROOT, 'save_model/50epoch/mnist_lenet5_AdaptiveMixedAT_0.5_5_3.pth')

    std_model = load_model(std_model_path)
    mix_model = load_model(mix_ig_pgd_path)

    # 加载测试数据
    imgs, lbls = get_test_data()

    # 参数设置
    eps = 0.1
    pgd_step = 20
    pgd_step_size = 0.025

    # 白盒攻击结果
    print('\n=== 白盒攻击（Mix-AT模型自身）===')

    # FGSM 白盒
    fgsm_wb = LinfPGD(net=mix_model, eps=eps, step=1, step_size=eps, random_start=False)
    x_adv_fgsm_wb = fgsm_wb((imgs, lbls))
    fgsm_wb_acc = test_accuracy(mix_model, x_adv_fgsm_wb, lbls)
    print(f'FGSM-WB: {fgsm_wb_acc:.2f}%')

    # PGD 白盒
    pgd_wb = LinfPGD(net=mix_model, eps=eps, step=pgd_step, step_size=pgd_step_size, random_start=True)
    x_adv_pgd_wb = pgd_wb((imgs, lbls))
    pgd_wb_acc = test_accuracy(mix_model, x_adv_pgd_wb, lbls)
    print(f'PGD-WB: {pgd_wb_acc:.2f}%')

    # CW 白盒
    cw_wb = LinfPGD(net=mix_model, eps=eps, step=pgd_step, step_size=pgd_step_size, random_start=True, criterion=CWLoss)
    x_adv_cw_wb = cw_wb((imgs, lbls))
    cw_wb_acc = test_accuracy(mix_model, x_adv_cw_wb, lbls)
    print(f'CW-WB: {cw_wb_acc:.2f}%')

    # 迁移攻击结果（从标准模型生成对抗样本，测试Mix-AT模型）
    print('\n=== 迁移攻击（从Standard迁移到Mix-AT）===')

    # FGSM 迁移
    fgsm_tr = LinfPGD(net=std_model, eps=eps, step=1, step_size=eps, random_start=False)
    x_adv_fgsm_tr = fgsm_tr((imgs, lbls))
    fgsm_tr_acc = test_accuracy(mix_model, x_adv_fgsm_tr, lbls)
    print(f'FGSM-Tr: {fgsm_tr_acc:.2f}%')

    # PGD 迁移
    pgd_tr = LinfPGD(net=std_model, eps=eps, step=pgd_step, step_size=pgd_step_size, random_start=True)
    x_adv_pgd_tr = pgd_tr((imgs, lbls))
    pgd_tr_acc = test_accuracy(mix_model, x_adv_pgd_tr, lbls)
    print(f'PGD-Tr: {pgd_tr_acc:.2f}%')

    # CW 迁移
    cw_tr = LinfPGD(net=std_model, eps=eps, step=pgd_step, step_size=pgd_step_size, random_start=True, criterion=CWLoss)
    x_adv_cw_tr = cw_tr((imgs, lbls))
    cw_tr_acc = test_accuracy(mix_model, x_adv_cw_tr, lbls)
    print(f'CW-Tr: {cw_tr_acc:.2f}%')

    # 干净样本准确率
    clean_acc = test_accuracy(mix_model, imgs, lbls)
    print(f'\nClean: {clean_acc:.2f}%')

    print('\n=== 结果汇总 ===')
    print(f'Mix-AT(Adaptive-IG+PGD): Clean={clean_acc:.2f}, FGSM-WB={fgsm_wb_acc:.2f}, FGSM-Tr={fgsm_tr_acc:.2f}, PGD-WB={pgd_wb_acc:.2f}, PGD-Tr={pgd_tr_acc:.2f}, CW-WB={cw_wb_acc:.2f}, CW-Tr={cw_tr_acc:.2f}')

if __name__ == '__main__':
    main()