# ========== 读取并显示白盒/迁移攻击对比表格 ==========
"""
读取 whitebox_vs_transfer_attack_comparison.csv 并显示为格式化表格
"""

import pandas as pd
import os

# 读取CSV
csv_path = './results_figures/whitebox_vs_transfer_attack_comparison.csv'

if not os.path.exists(csv_path):
    print(f'[ERROR] CSV文件不存在: {csv_path}')
    print('请先运行白盒攻击+迁移攻击测试cell生成数据')
else:
    # 读取数据
    df = pd.read_csv(csv_path)

    # 设置显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)

    print('=' * 100)
    print('白盒攻击 vs 迁移攻击 对比结果')
    print('=' * 100)
    print()

    # 显示完整表格
    print(df.to_string(index=False))

    print()
    print('=' * 100)
    print(f'数据行数: {len(df)} | 模型数: {df["model"].nunique()} | 攻击类型数: {df["attack_type"].nunique()}')
    print('=' * 100)
