# -*- coding: utf-8 -*-
"""DeepFish_MaskRCNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T5iP9rMYHws_IehzpafvBWR1B-l4HpNa

# 1. **Installation**
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
!pip install --upgrade h5py==2.10.0
!wget https://www.dropbox.com/s/nne2h2e1ddmpv5k/Mask_RCNN_basic_1.zip
!unzip Mask_RCNN_basic_1.zip
import sys
sys.path.append("/content/Mask_RCNN/mrcnn")
from Mask_RCNN.mrcnn import *
# %matplotlib inline

!nvidia-smi

"""# 2. **Image dataset**"""

# Extract Images

images_path = "trainingDataset.zip"
annotations_path = "annotations.json"

extract_images(os.path.join("/content/",images_path), "/content/trainDataset")

dataset_train = load_image_dataset(os.path.join("/content/", annotations_path), "/content/trainDataset", "train")
dataset_val = load_image_dataset(os.path.join("/content/", annotations_path), "/content/trainDataset", "val")
class_number = dataset_train.count_classes()
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))

# Load image samples
display_image_samples(dataset_train)

"""# 3. Training"""

# Load Configuration
config = CustomConfig(class_number)
#config.display()
model = load_training_model(config)

# Start Training
# This operation might take a long time.
#train_head(model, dataset_train, dataset_train, config)

## or you can directly download the model without re-train from here:
!wget https://www.dropbox.com/s/evfvlt3eitqpnkr/mask_rcnn_object_0005.h5?dl=0

"""# 4. **Detection**"""

# Load Test Model
# The latest trained model will be loaded
test_model, inference_config = load_test_model(class_number)

# Test on a random image
test_random_image(test_model, dataset_val, inference_config)

"""# 5. **Use the mask RCNN on random (underwater) images**"""

from visualize import random_colors, get_mask_contours, draw_mask

# Load Image
img = cv2.imread("/content/9898_Acanthopagrus_palmaris_f000050.jpg")

test_model, inference_config = load_inference_model(1, "/content/mask_rcnn_object_0005.h5")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detect results
r = test_model.detect([image])[0]
colors = random_colors(80)

from google.colab.patches import cv2_imshow
# Get Coordinates and show it on the image
object_count = len(r["class_ids"])
for i in range(object_count):
    # 1. Mask
    mask = r["masks"][:, :, i]
    contours = get_mask_contours(mask)
    for cnt in contours:
        cv2.polylines(img, [cnt], True, colors[i], 2)
        img = draw_mask(img, [cnt], colors[i])

cv2_imshow(img)
