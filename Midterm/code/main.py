import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import argparse

from models import *
import utils


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

utils.set_seed(42)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# parameter setting
best_acc = 0  
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
epochs = 50

# %%
# Data Loader
print('==> Preparing data..')

# extract image paths
tr_image_paths, tr_labels = utils.load_path(filepath='Midterm/data/train', train=True)
ts_image_paths, _ = utils.load_path(filepath='Midterm/data/test_3000_nolabel', train=False)

print("Train size: {}\nTest size: {}".format(len(tr_image_paths), len(ts_image_paths)))

class_to_idx = {}
idx = 0
for label in tr_labels:
    if label not in class_to_idx:
        class_to_idx[label] = idx
        idx += 1
idx_to_class = {value:key for key,value in class_to_idx.items()}

tr_transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

ts_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# encode for dataloader
tr_dataset = utils.LandmarkDataset(tr_image_paths, class_to_idx, tr_transform)
ts_dataset = utils.LandmarkDataset(ts_image_paths, None, ts_transform)

tr_loader = torch.utils.data.DataLoader(
    tr_dataset, batch_size=batch_size, shuffle=True)#, num_workers=2)
ts_loader = torch.utils.data.DataLoader(
    ts_dataset, batch_size=batch_size, shuffle=False)#, num_workers=2)

# %%
# Model
print('==> Building model..')
# net = SimpleDLA(num_classes=len(class_to_idx))
net = Simple(num_classes=len(class_to_idx))
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('Midterm/result/result.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# %% 
# Training
def train(epoch):
    global best_acc
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tr_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(batch_idx, len(tr_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        torch.save(state, 'Midterm/result/result.pth')
        best_acc = acc

'''
for epoch in range(start_epoch, start_epoch+epochs):
    train(epoch)
    scheduler.step()
'''
# %%
def test():
    net.eval()
    test_idxes = []
    test_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(ts_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            test_idxes.append(predicted)

    test_idxes = torch.cat(test_idxes)

    for idx in test_idxes:
        test_labels.append(int(idx_to_class[int(idx)]))

    return test_labels

net.load_state_dict(torch.load('Midterm/result/result.pth',  map_location=device)['net'])
test_id = pd.read_csv('Midterm/result/results_samples.csv')['imagename']
test_labels = test()

result_df = pd.DataFrame(
    {'imagename': test_id,
     'predicted': test_labels
    })

result_df.to_csv("Midterm/result/results.csv", index=False)