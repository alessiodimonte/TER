#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow-gpu==1.15.0')
get_ipython().system('pip install --upgrade h5py==2.10.0')


# In[ ]:


import sys
sys.path.append("Mask_RCNN_withRW/mrcnn")


# In[ ]:


from m_rcnn import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## Create zip file for the images to consider for our training set

import shutil

# copy images from test to the images of train
get_ipython().system('cp -r "../testImages" "../trainImages" ##copy the folder testImages into trainImages')
get_ipython().system('cp -r "../trainImages/testImages/" "../trainImages" ##copy the content of the folder testImages into the folder trainImages')

get_ipython().system('rm -r "../trainImages/testImages" ##remove the folder testImages copied inside trainImages')

# create zip file
shutil.make_archive("trainingDataset", 'zip', "../trainImages")


# In[ ]:


## Extract Images

images_path = "trainingDataset.zip"
annotations_path = "../trainAnnotations/annotations.json"

extract_images(os.path.join("",images_path), "dataset")


# In[ ]:


dataset_train = load_image_dataset(os.path.join("", annotations_path), "dataset", "train")
dataset_val = load_image_dataset(os.path.join("", annotations_path), "dataset", "val")

class_number = dataset_train.count_classes()
print('Train: %d' % len(dataset_train.image_ids))
print('Validation: %d' % len(dataset_val.image_ids))
print("Classes: {}".format(class_number))


# In[ ]:


# Load image samples
display_image_samples(dataset_train)


# In[ ]:


# Load Configuration
config = CustomConfig(class_number)

#config.display()
model = load_training_model(config)


# TRAINING:

# In[ ]:


# Start Training
# This operation might take a long time.
train_head(model, dataset_train, dataset_train, config)


# TESTING:

# In[ ]:


from visualize import random_colors, get_mask_contours, draw_mask


# In[ ]:


import os

list = os.listdir("../testImages") # dir is your directory path
number_files = len(list)
print(number_files)


# In[ ]:


loc = "../testImages" 

names=[] 

for images in os.listdir(loc): 
    file = loc+"/"+images
    filestr = file.split("/")
    
    names.append(filestr[2])


# In[ ]:


get_ipython().system('pip install opencv-python')


# In[ ]:


## if you want to infer results from the already trained model you have to download the .h5 file (which we uploaded on DropBox)

get_ipython().system('wget "https://www.dropbox.com/s/evfvlt3eitqpnkr/mask_rcnn_object_0005.h5"')


# In[ ]:


import cv2
import numpy as np


# Load Image
for images in range(number_files):
    
    img = cv2.imread("../testImages/names[images]")

    test_model, inference_config = load_inference_model(1, "/mask_rcnn_object_0005.h5")
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect results
    r = test_model.detect([image])[0]
    colors = random_colors(80)
    
    ## Save picture with colored mask in the folder "outputImagesWithMasksColors"
    object_count = len(r["class_ids"])

    for i in range(object_count):
        # 1. Mask
        mask = r["masks"][:, :, i]
        contours = get_mask_contours(mask)

        for cnt in contours:
            cv2.polylines(img, [cnt], True, colors[i], 2)
            img = draw_mask(img, [cnt], colors[i])

        cv2.imwrite(os.path.join("../outputImagesWithMasksColors" , 'names[images]'), img)
    
    ## Create the mask in black and white
    height, width, channels = image.shape
    image_shape = height, width
    
    ## We want a black and white image to pass to the annotation script
    black_image = np.zeros((height,width,3), np.uint8)
    white = (255, 255, 255)
    
    ## Save the mask in B&W in the folder "outputImagesWithMasksBlackAndWhite"
    object_count = len(r["class_ids"])
    for i in range(object_count):
        # 1. Mask
        mask = r["masks"][:, :, i]
        contours = get_mask_contours(mask)
        for cnt in contours:
            img2 = draw_mask(black_image, [cnt], white)
            cv2.polylines(black_image, [cnt], True, white, 2)
            cv2.fillPoly(black_image, [cnt], white)
        
        cv2.imwrite(os.path.join("../outputImagesWithMasksColors" , 'names[images]'), img2)


# RUN THE MASK2COCO SCRIPT IN ORDER TO CREATE THE ANNOTATION FILE:

# In[ ]:


get_ipython().system('ipython ../mask2COCO/mask_to_coco_converter.py')


# In[ ]:


## now we can delete the images contained in testImages folder (not the folder itself) because they became part of the trainImages
get_ipython().system('rm ../testImages/*')


# In[ ]:


## now we can delete the .h5 from the local directory
get_ipython().system('rm mask_rcnn_object_0005.h5')

