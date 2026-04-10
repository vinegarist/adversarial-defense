"""
多模型对比可视化 - 两张图
使用方式:
    from plot_multi_model_comparison import plot_comparison_figures
    plot_comparison_figures(df_comparison_full, N, R)
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_figures(df_comparison_full, N, R):
    """绘制两张多模型对比图

    图1: 本次实验的遮蔽模型 vs 遮蔽攻击
    图2: 所有模型 vs 所有攻击（包含IG和Occ攻击）
    """
    if df_comparison_full is None or df_comparison_full.empty:
        print('[WARN] 未找到 df_comparison_full，请先运行多模型对比评测')
        return

    # ========== 图1: 本次实验的遮蔽模型 vs 遮蔽攻击 ==========
    occlusion_models = ['Adaptive-Saliency-AT', 'Adaptive-Saliency-AT(N=5,R=3)',
                        'Mix-AT', 'Adaptive-Mix-AT', 'Standard']
    occlusion_attacks = ['Clean', 'Fixed-Saliency(k=3)', 'Fixed-Saliency(k=9)',
                         'Fixed-Saliency(k=15)', 'Adaptive-Saliency(N=5,R=3)',
                         'Adaptive-Saliency(N=10,R=3)']

    available_occlusion_models = [m for m in occlusion_models if m in df_comparison_full['Model'].values]
    available_occlusion_attacks = [a for a in occlusion_attacks if a in df_comparison_full.columns]

    if len(available_occlusion_models) > 0 and len(available_occlusion_attacks) > 0:
        fig1, ax1 = plt.subplots(figsize=(14, 7))

        x = np.arange(len(available_occlusion_attacks))
        width = 0.8 / len(available_occlusion_models)
        colors = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'mediumpurple']

        for i, model_name in enumerate(available_occlusion_models):
            row = df_comparison_full[df_comparison_full['Model'] == model_name].iloc[0]
            values = [row.get(at, 0) for at in available_occlusion_attacks]
            offset = width * (i - len(available_occlusion_models)/2 + 0.5)
            bars = ax1.bar(x + offset, values, width, label=model_name,
                          color=colors[i % len(colors)], alpha=0.85, edgecolor='black', linewidth=1)

            for bar, val in zip(bars, values):
                if val > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8, rotation=0)

        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('本次实验: 遮蔽攻击防御模型对比\n(Adaptive-Saliency-AT / Mix-AT / Adaptive-Mix-AT)', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(available_occlusion_attacks, rotation=15, ha='right')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 105)

        plt.tight_layout()
        save_path1 = f'./results_figures/occlusion_models_vs_occlusion_attacks_{N}_{R}.png'
        plt.savefig(save_path1, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'遮蔽模型对比图已保存: {save_path1}')

    # ========== 图2: 所有模型 vs 所有攻击 ==========
    all_models = ['Standard', 'PGD-AT', 'FGSM-AT', 'Occlusion-AT', 'Mix-AT(Occlusion+PGD)',
                  'Adaptive-Saliency-AT', 'Adaptive-Saliency-AT(N=5,R=3)',
                  'Adaptive-Occlusion-AT', 'Mix-AT', 'Adaptive-Mix-AT']
    all_attacks = ['Clean', 'FGSM', 'PGD', 'CW',
                   'Fixed-Saliency(k=3)', 'Fixed-Saliency(k=9)', 'Fixed-Saliency(k=15)',
                   'Adaptive-Saliency(N=5,R=3)', 'Adaptive-Saliency(N=10,R=3)',
                   'IG-Fixed(k=9)', 'IG-Adaptive(N=5,R=3)', 'Occ-Fixed(k=9)']

    available_all_models = [m for m in all_models if m in df_comparison_full['Model'].values]
    available_all_attacks = [a for a in all_attacks if a in df_comparison_full.columns]

    if len(available_all_models) > 0 and len(available_all_attacks) > 0:
        fig2, ax2 = plt.subplots(figsize=(18, 8))

        x = np.arange(len(available_all_attacks))
        width = 0.8 / len(available_all_models)
        colors_all = ['steelblue', 'coral', 'mediumseagreen', 'gold', 'mediumpurple',
                      'lightcoral', 'lightskyblue', 'lightgreen', 'plum', 'wheat']

        for i, model_name in enumerate(available_all_models):
            row = df_comparison_full[df_comparison_full['Model'] == model_name].iloc[0]
            values = [row.get(at, 0) if row.get(at, None) is not None else 0 for at in available_all_attacks]
            offset = width * (i - len(available_all_models)/2 + 0.5)
            bars = ax2.bar(x + offset, values, width, label=model_name,
                          color=colors_all[i % len(colors_all)], alpha=0.85, edgecolor='black', linewidth=0.5)

            for bar, val in zip(bars, values):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=7, rotation=90)

        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('完整对比: 所有模型 vs 所有攻击类型\n(包含Saliency、IG、Occlusion攻击)', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(available_all_attacks, rotation=30, ha='right')
        ax2.legend(fontsize=9, loc='upper right', ncol=2)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0, 110)

        plt.tight_layout()
        save_path2 = f'./results_figures/all_models_vs_all_attacks_comprehensive_{N}_{R}.png'
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'完整对比图已保存: {save_path2}')

    print('\n========== 可视化完成 ==========')
    print(f'图1: 遮蔽模型对比 - {len(available_occlusion_models)} 个模型, {len(available_occlusion_attacks)} 种攻击')
    print(f'图2: 完整对比 - {len(available_all_models)} 个模型, {len(available_all_attacks)} 种攻击')


if __name__ == '__main__':
    print("请使用: from plot_multi_model_comparison import plot_comparison_figures")
    print("然后调用: plot_comparison_figures(df_comparison_full, N, R)")
