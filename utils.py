import torch
import torchvision
from torchvision import transforms


def load_mnist_test(n_examples=None, download=False, root='./data/'):
    """
    我们并非每次都测试整个测试集的样本，本函数用于快速地加载指定个数的MNIST测试集样本

    参数说明：
    n_examples: 加载样本的个数。如果为None或大于总样本数，则加载所有样本；否则将加载
    MNIST测试集中前n_examples个样本与标签
    download: 是否下载数据集
    root: 数据集存放路径
    """
    batch_size = 100
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor