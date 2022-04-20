import torchvision.transforms as transforms
import numpy as np
import utils
import matplotlib.pyplot as plt


# extract image paths
tr_image_paths, tr_labels = utils.load_path(filepath='Midterm/data/train', train=True)

class_to_idx = {}
idx = 0
for label in tr_labels:
    if label not in class_to_idx:
        class_to_idx[label] = idx
        idx += 1
idx_to_class = {value:key for key,value in class_to_idx.items()}

tr_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()])
tr_dataset = utils.LandmarkDataset(tr_image_paths, class_to_idx, tr_transform)

# %%
# class distribution 
count_class = {key:0 for key in class_to_idx}
for _, y in tr_dataset:
    count_class[idx_to_class[y]] += 1

print(count_class)
plt.bar(count_class.keys(), count_class.values(), width=0.5, color='g')
plt.savefig('Midterm/result/class_distribution.png', dpi=300)
plt.clf()
# %%
# train data RGB mean, std
meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in tr_dataset]
stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in tr_dataset]

meanR = np.mean([m[0] for m in meanRGB])
meanG = np.mean([m[1] for m in meanRGB])
meanB = np.mean([m[2] for m in meanRGB])

stdR = np.mean([s[0] for s in stdRGB])
stdG = np.mean([s[1] for s in stdRGB])
stdB = np.mean([s[2] for s in stdRGB])

print(meanR, meanG, meanB) # 0.29392457 0.30366316 0.29798967
print(stdR, stdG, stdB) # 0.15284057 0.14239912 0.14335868
