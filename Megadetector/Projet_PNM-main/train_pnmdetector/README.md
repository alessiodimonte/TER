# Train PNMdetector

## 1. Installation of Tensorflow 1 Object detection API

Install version of TFOD API 1.12.0, inspired by: https://github.com/microsoft/CameraTraps/tree/master/detection

For the installation, in addition to following instructions follow this tutorial: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/install.html

Use python 3.6 and create two environments conda, one for CPU, one for GPU.

### a) Installation python + TF cpu or gpu

conda create -n pyt36tf112cpu python=3.6

conda activate pyt36tf112cpu 

pip install tensorflow ==1.12 (cpu)

or pip install tensorflow-gpu==1.12 (gpu)

pip install pillow lxml jupyter matplotlib cython opencv-python humanfriendly tqdm jsonpickle statistics requests pandas sklearn

### b) Download TF models

Download: https://github.com/tensorflow/models/tree/r1.13.0 or here models-r1.13.0: https://unice-my.sharepoint.com/:f:/g/personal/fanny_simoes_unice_fr/EsNTzYUEyb1NroMUuxMHt5EBkFKigSEBgICBJe4tibDfRw?e=wefhh2

And save TFmodel : {yourlink}/tensorflow1/

### c) Coco API installation

pip install pycocotools (for pyt36tf112cpu and pyt36tf112gpu)

### d) Protobuf installation/compilation

brew install protobuf 

or 

Download: https://github.com/protocolbuffers/protobuf/releases and save protobuf

cd {yourlink}/tensorflow1/models/research/

export PATH=$PATH: {yourlink_save_protobuf}/protobuf/bin

protoc object_detection/protos/*.proto --python_out=.

### e) Install object detection

cd {yourlink}/tensorflow1/models/research/

pip install .

## 2. Train model

In addition to following instructions, follow this tutorial: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/training.html 

### a) Import MegaDetector Model v4.1

- https://github.com/microsoft/CameraTraps/blob/master/megadetector.md (last checkpoint) or you can download here megadetector: https://unice-my.sharepoint.com/:f:/g/personal/fanny_simoes_unice_fr/EsNTzYUEyb1NroMUuxMHt5EBkFKigSEBgICBJe4tibDfRw?e=wefhh2

- Import in file "pre-trained-models"

### b) Data annotated

- Links to images with labels and bounding boxes associated: sample_imgBB.json
- Be careful: the links of images have to be modified according to the location of images.

### c) Creating Label Map

- Use: label_map.pbtxt
- At the moment, 13 species are selected until "bouquetin".
- If you selected less or more species you have to modificate this file.

### d) Partitioning the images

- Use script: train-val-test_uniqueimg.py
- Train (80\%) Validation (10\%) Test (10\%).
- scripts classify images (train-validation-test) in file "images" and create csv files save in "annotations" for each part which contains names of images, labels, BB coordinates of MegaDetector (carefull to format see description in their website github), width, height and real coordinates of BB.
- Images are unique by train-validation-test.

### e) Creating Tensorflow Records

- Use script: generate_tfrecord_train-val-test_uniqueimg.py 
- Transform csv files save in "annotations" to record files in "annotations".

### f) Configuring training pipeline

- Use: pnmdetector.config
- To fix parameters of model: number of classes, links to images (train and validation), link to ckpt file of MegaDetector model and  link to labelmap.
- Possibility to change others parameters. 

### g) Train model

- Use: model_main_megadetector.py
- Possibility to change some values like save_checkpoints_steps.

### h) Visualize results with Tensorboard

- Visualize mAP, loss function, AR, evaluation.

conda activate pyt36tf112cpu

Tensorboard – logdir= link_to_model_train --host localhost

## 3. Export model trained

- Follow this tutorial: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/training.html#exporting-a-trained-inference-graph
- Select your best model according to ckpt files
- Use script: export_inference_graph.py like in tutorial
