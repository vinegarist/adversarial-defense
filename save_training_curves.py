# 保存训练曲线数据到CSV的辅助脚本
# 用于补充已有notebook的CSV保存功能

import pandas as pd

# Adaptive-Saliency-AT模型的训练曲线保存
def save_adaptive_at_training_curves(train_losses, train_accs, test_losses, test_accs, test_clean_accs, N, R, EPOCH):
    """保存Adaptive-Saliency-AT训练曲线到CSV"""
    df_training_history = pd.DataFrame({
        'epoch': list(range(1, EPOCH+1)),
        'train_loss': train_losses,
        'train_acc': [100*a for a in train_accs],
        'test_loss': test_losses,
        'test_acc': [100*a for a in test_accs],
        'test_clean_acc': [100*a for a in test_clean_accs]
    })
    csv_path = f'./results_figures/adaptive_saliency_at_training_history_{N}_{R}.csv'
    df_training_history.to_csv(csv_path, index=False)
    print(f'训练曲线数据已保存: {csv_path}')
    return csv_path

# Mix-AT模型的训练曲线保存
def save_mix_at_training_curves(train_losses_mix, train_accs_mix, test_losses_mix, test_accs_mix, test_clean_accs_mix,
                                 N_MIX, R_MIX, OCCLU_RATIO_MIX, EPOCH_MIX):
    """保存Mix-AT训练曲线到CSV"""
    df_training_history = pd.DataFrame({
        'epoch': list(range(1, EPOCH_MIX+1)),
        'train_loss': train_losses_mix,
        'train_acc': [100*a for a in train_accs_mix],
        'test_loss': test_losses_mix,
        'test_acc': [100*a for a in test_accs_mix],
        'test_clean_acc': [100*a for a in test_clean_accs_mix]
    })
    csv_path = f'./results_figures/mix_at_training_history_{N_MIX}_{R_MIX}_{OCCLU_RATIO_MIX}.csv'
    df_training_history.to_csv(csv_path, index=False)
    print(f'Mix-AT训练曲线数据已保存: {csv_path}')
    return csv_path

if __name__ == '__main__':
    print("训练曲线保存函数已定义，可在notebook中导入使用:")
    print("from save_training_curves import save_adaptive_at_training_curves, save_mix_at_training_curves")
