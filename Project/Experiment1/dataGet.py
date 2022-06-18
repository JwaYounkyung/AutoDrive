'''
Settings
!rm -rf waymo-od > /dev/null
!git clone https://github.com/waymo-research/waymo-open-dataset.git waymo-od
!cd waymo-od && git branch -a
!cd waymo-od && git checkout remotes/origin/master
!pip3 install --upgrade pip
!pip3 install waymo-open-dataset-tf-2-1-0==1.2.0
'''

import os
import tensorflow.compat.v1 as tf

import matplotlib.pyplot as plt
import matplotlib.patches as patches

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as open_dataset

FILENAME = 'waymo-od/tutorial/frames'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    break

print(frame.context)

def show_camera_image(camera_image, camera_labels, cmap=None):
  plt.figure()
  ax = plt.subplot()

  for camera_labels in frame.camera_labels:
    if camera_labels.name != camera_image.name:
      continue

    for label in camera_labels.labels:
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
            width=label.box.length,
            height=label.box.width,
            linewidth=1,
            edgecolor='red',
            facecolor='none'))

  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')
  plt.savefig('Experiment1/result/origin_waymo.png')

def camera_image_save(camera_image, cmap=None):
  plt.figure()
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.grid(False)
  plt.axis('off')
  plt.savefig('Experiment1/result/waymo.png')

for index, image in enumerate(frame.images):
  camera_image_save(image)
  show_camera_image(image, frame.camera_labels)
  break