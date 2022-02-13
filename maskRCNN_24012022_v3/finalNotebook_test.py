#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("Mask_RCNN_withRW/mrcnn")
from m_rcnn import *
get_ipython().run_line_magic('matplotlib', 'inline')


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


# In[ ]:


get_ipython().system('ipython ../mask2COCO/mask_to_coco_converter.py')

