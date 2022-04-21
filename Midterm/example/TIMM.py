# How to use pretrained model
# https://rwightman.github.io/pytorch-image-models/models/vision-transformer/

import timm
from pprint import pprint

import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import torch
import torchvision.transforms as transforms

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=6)
model.eval()

config = resolve_data_config({}, model=model)
transform = create_transform(**config)
tr_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()]
)
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)
img = Image.open(filename).convert('RGB')
tensor = tr_transform(img).unsqueeze(0) # transform and add batch dimension

with torch.no_grad():
    out = model(tensor)
probabilities = torch.nn.functional.softmax(out[0], dim=0)
print(probabilities.shape)
# prints: torch.Size([1000])