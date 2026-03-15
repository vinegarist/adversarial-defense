import os
import gzip
import struct
import torch
import numpy as np

raw_dir = os.path.join('.', 'data', 'MNIST', 'raw')
proc_dir = os.path.join('.', 'data', 'MNIST', 'processed')
if not os.path.exists(proc_dir):
    os.makedirs(proc_dir)

def read_images(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            data = f.read()
    else:
        with open(path, 'rb') as f:
            data = f.read()
    magic = struct.unpack('>I', data[:4])[0]
    if magic != 2051:
        raise ValueError('Invalid magic for images: %s' % magic)
    num, rows, cols = struct.unpack('>III', data[4:16])
    imgs = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(num, rows, cols)
    return torch.from_numpy(imgs)

def read_labels(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            data = f.read()
    else:
        with open(path, 'rb') as f:
            data = f.read()
    magic = struct.unpack('>I', data[:4])[0]
    if magic != 2049:
        raise ValueError('Invalid magic for labels: %s' % magic)
    num = struct.unpack('>I', data[4:8])[0]
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return torch.from_numpy(labels).long()

candidates = {
    'train_img': ['train-images-idx3-ubyte.gz', 'train-images-idx3-ubyte'],
    'train_lbl': ['train-labels-idx1-ubyte.gz', 'train-labels-idx1-ubyte'],
    'test_img': ['t10k-images-idx3-ubyte.gz', 't10k-images-idx3-ubyte'],
    'test_lbl': ['t10k-labels-idx1-ubyte.gz', 't10k-labels-idx1-ubyte']
}

found = {}
for k, names in candidates.items():
    for n in names:
        p = os.path.join(raw_dir, n)
        if os.path.exists(p):
            found[k] = p
            break
    if k not in found:
        print('Missing', k, 'expected one of', names)
        raise SystemExit(1)

print('Found raw files:', found)

train_images = read_images(found['train_img'])
train_labels = read_labels(found['train_lbl'])
test_images = read_images(found['test_img'])
test_labels = read_labels(found['test_lbl'])

# save as torchvision processed format: training.pt and test.pt
train_path = os.path.join(proc_dir, 'training.pt')
test_path = os.path.join(proc_dir, 'test.pt')

print('Saving', train_path)
torch.save((train_images, train_labels), train_path)
print('Saving', test_path)
torch.save((test_images, test_labels), test_path)

print('Done. Processed files:', os.listdir(proc_dir))
