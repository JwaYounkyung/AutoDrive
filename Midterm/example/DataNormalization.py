# 이미지 픽셀의 평균, 표준편차를 계산하여 정규화 하기
# https://deep-learning-study.tistory.com/475

from torchvision import datasets
import torchvision.transforms as transforms
import os

import numpy as np
import torch

import matplotlib.pyplot as plt

path2data = 'data/'

data_transformer = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.STL10(path2data, split='train', download='True', transform=data_transformer)

print(train_ds.data.shape)

meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in train_ds]
stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in train_ds]

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print(meanR, meanG, meanB)
print(stdR, stdG, stdB)

train_transformer = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
])