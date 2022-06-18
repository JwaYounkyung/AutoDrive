import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torchvision.transforms as T
import tensorflow.compat.v1 as tf
torch.set_grad_enabled(False);
tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset
from model import DETRdemo

# %%

# COCO classes (91)
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform= T.Compose([
    T.Resize(800), 
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    img = transform(im).unsqueeze(0)
    print(img.shape)
    if img.shape[2] > 1600:
        img = img[:,:,:1600]
    if img.shape[3] > 1600:
        img = img[:,:,:,:1600]
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600

    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

#visualize
def plot_results(pil_img, prob, boxes, idx):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.savefig('Experiment3/result/waymo' + str(idx) + '.png')

def plot_results_combine(pil_img, prob, boxes, camera_image, camera_labels, idx):
    plt.figure()
    plt.imshow(pil_img)

    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    
    ax2 = plt.subplot()
    for camera_labels in frame.camera_labels:
        if camera_labels.name != camera_image.name:
            continue

        for label in camera_labels.labels:
            ax2.add_patch(patches.Rectangle(
                xy=(label.box.center_x - 0.5 * label.box.length,
                    label.box.center_y - 0.5 * label.box.width),
                    width=label.box.length,
                    height=label.box.width,
                    linewidth=2,
                    edgecolor='black',
                    facecolor='none'))
    
    
    plt.axis('off')
    plt.savefig('Experiment3/result/waymo_combine' + str(idx) + '.png')
# %% 
# data preprocessing
FILENAME = 'waymo-od/tutorial/frames'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

frames = []
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    frames.append(frame)

# %%
# model setting
detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval()

# %%
# Inference
start = time.time()

idx = 1
for frame in frames:
    for i, camera_image in enumerate(frame.images):
        image = tf.image.decode_png(camera_image.image).numpy()
        image = Image.fromarray(image)

        scores, boxes = detect(image, detr, transform)
        plot_results(image, scores, boxes, idx)
        plot_results_combine(image, scores, boxes, camera_image, frame.camera_labels, idx)
        idx += 1

print("Inference time :", round(time.time()-start, 3), 'sec')