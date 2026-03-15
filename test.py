import torch
import math


@torch.no_grad()
def test(classifier, samples, labels, bs=100, mode='clean'):
    """
    参数说明：
    classifier: 待测试的分类器
    samples: 待测试的样本
    labels: 样本标签
    bs: 测试时的batch_size
    mode: 测试模型，支持的选项列表：['attack', 'clean']

    对于本函数来说，classifier只要返回的是类似于logits或概率的向量即可，其分类过程并不
    重要。换句话说，它可以是一个简单的cnn，也可以是包含先攻击后分类两个过程以测试鲁棒性。
    很快，我们将看到如何使用此函数快速地执行FGSM, PGD, CW测试。

    返回值说明：
    准确率, 预测列表
    """
    n = samples.shape[0]
    correct = 0
    lens = math.ceil(n / bs)
    count = 0

    preds = []
    for i in range(lens):
        idx_b = i * bs
        idx_e = min((i + 1) * bs, n)
        x = samples[idx_b:idx_e].cuda()
        y = labels[idx_b:idx_e].cuda()

        if mode == 'attack':
            # 包含攻击流程的分类器需要 y（ground truth）来执行攻击
            pred = classifier((x, y)).argmax(-1)
        elif mode == 'clean':
            pred = classifier(x).argmax(-1)
        else:
            raise NotImplementedError
        
        preds.append(pred.cpu())
        correct += pred.eq(y).sum().item()
        count += y.shape[0]

    preds = torch.cat(preds)

    return 100 * correct / n, preds
    