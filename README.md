# Introduction
## Problem
One of the greatest humans’ abilities is the detection and classification of objects in visual scenarios (real life, images, and videos) with a natural accuracy, simplicity, and velocity. Humans preserve this ability also for underwater ecosystems, being able to easily discriminate the position of fishes, corals, and marine-related objects. Beyond that, humans with a certain expertise in the marine field are capable to distinguish different species, classifying each object in the correct way.

What is required in the development of the project is to automatically detect and recognize fishes in the marine ecosystem without passing through experts each time it is necessary to perform this kind of task. In fact, in the case the process needs to be done at a high scale and considering many species, it might become a process requiring a notable amount of time. Fish experts should spend a substantial amount of their time drawing the shape and assigning the label for each fish. On a practical side, it is clearly an ineffective use of time, energy, and money. Considering than on average 2 minutes are required for annotating a single image by an expert, that means that the annotation of 1000 images would require more than 33 hours. This amount of time turns out to be a waste of the costly experts’ working hours.

To tackle this problem a machine learning solution can be deployed to accomplish the same duty in an automatic way, saving in this way time for experts who can dedicate themselves to other activities.

## Context and users

The users involved are the experts from the Computer Science, Signals and Systems laboratory of Sophia Antipolis (I3S) and the Ecology and Conservation Science for Sustainable Seas (ECOSEAS), whose research interest is biology and who are focusing on the monitoring and safeguarding of the Mediterranean area, with the aim of protecting the biodiversity of the sea [1][2].

The main outcome of the project is to support the biologists with an efficient tool they can exploit for fish location and class extraction, with the ability of potentially perform large-scale operations on many images.

This is the reason why a tool for the analysis and monitoring of the Mediterranean Sea area is considered as a solid starting point for the above-mentioned players.

## Scope qualification
Considering the dataset dimension, the project aims to use images representing the underwater ecosystem within the Mediterranean Sea by using the SeaCLEF dataset [3]. However, due to its unavailability, the DeepFish dataset was used instead. This dataset contains approximately 40 thousand images of fishes, and because of that it enables the recognition of fishes from different scenarios, allowing its use in more general contexts [4].

For the technological aspect, the software created can identify, locate, and create a mask for fishes in different images classifying them in a unique “Fish” class. Regarding the accuracy of the model, it relies on the number of images annotated in the dataset, which can be improved by manual annotation from the experts through the User Interface.

## References
[1]	Computer Science, Signals and Systems Laboratory of Sophia Antipolis, https://www.i3s.unice.fr/

[2]	Ecology and Conservation Science for Sustainable Seas, http://ecoseas.unice.fr/

[3]	SeaCLEF, 2017, https://www.imageclef.org/lifeclef/2017/sea 

[4]	Alxayat Saleh, Issam H. Laradji, Dmitry A. Konovalov, Michael Bradley, David Vazquez, Marcus Sheaves, 2020, “A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis”, Arxiv, https://arxiv.org/abs/2008.12603


# Structure of the Github Repository
In the **master** branch it is stored the code for the working User Interface, comprehending all the parts for running the "Mask R-CNN" training and test phase and the "Flask" code for the clickable button.

In the **google-colab** branch it is stored the code adapted to work on Google Colab in case the local machine gives problems.

In the **archive** branch it is stored the code for all the different parts studied throughout the development of the project, organized in single folders divided by topic.


# User Manual 
## Local Machine 
### 1. Install the UI 
1.1. Download from the GitHub repository the branch “master”. 
  - In the terminal window write 
    
      ``git clone https://github.com/alessiodimonte/TER.git``
      
      
      ![image](https://user-images.githubusercontent.com/23140351/153864271-39275bd0-a8bf-4cd0-ab33-afa5412b7ef5.png)
      
      
1.2. Download and install node.js version 12.x.x and npm version 6.14.x (other version might not work with the UI) (for example: https://nodejs.org/download/release/v12.22.8). 

1.3. Open the terminal in the just downloaded folder in the path "TER/annotationTool"
  - For example, if you have donwloaded the folder on the Desktop, in the terminal you need to write 
      ``cd Desktop/TER/annotationTool``
      
1.4. In the same terminal window write `` npm install``

1.5. In the same terminal window write `` npm rebuild node-sass``

1.6. In the same terminal window write `` npm start``

1.7. The UI will be automatically opened in a browser tab (the process might take several minutes), in the case nothing is opened just write on the browser "http://localhost:3000/"

![image](https://user-images.githubusercontent.com/23140351/153865004-6036e67d-3d6d-4fb6-b409-eb8a7bfc02b6.png)


### 2. Use the UI

2.1. Click on the "Drop images or click here to select them" button and select the images you want to annotate

2.2. Click on the "Object Detection" button

2.3. Define label(s) by either click on the "+" or "Load labels from file" button

2.4. Click "Start project" button

2.5. Load the annotations with the buttons "Actions" &rarr; "Import Annotations" and select the annotation file corresponding to the images uploaded in step 2.1 (or make annotations by yourself)
  - The images with the tick icon means that are annotated, the ones with the forbid icon are not

![image](https://user-images.githubusercontent.com/23140351/153866317-398be19f-41ce-488e-8bc3-85849f192587.png)

2.6. From the UI it is possible to create new annotations and change the imported annotations masks

2.7. It is posible to export the annotation by clicking "Actions" &arr "Export Annotations" &arr "COCO JSON"

![image](https://user-images.githubusercontent.com/23140351/153866535-1cc2f91d-c416-4b3d-a481-478ff5f5a077.png)

### 3. Training and Testing on the Local Machine

3.1. Open a new terminal window in the folder "TER/maskRCNN_24012022_v3"

![image](https://user-images.githubusercontent.com/23140351/153866657-acfa8c8c-5bb8-4960-b23c-68fa460e4b36.png)

3.2. In the same terminal window write 

  `` python startTrainingFlask.py``
  
3.3. In the folder "TER/testImages" put the images on which you want to automatically create your annotations

![image](https://user-images.githubusercontent.com/23140351/153866843-ef3b0aaf-ddca-4e72-8557-b1803dcbe3af.png)

3.4. Return to the UI and click on the "Start Training" button in the UI

![image](https://user-images.githubusercontent.com/23140351/153866926-ffa429a2-57ea-49e1-9182-721c244f7849.png)

3.5. The script is run in background and saves the annotation file in the folder "testAnnotations"


## Google Colab (in the case the training and/or testing does not work on the local machine)

### 4. Training and Testing in Google Colab

4.1. In a terminal window write 
  `` git clone google-colab https://github.com/alessiodimonte/TER.git``
  
  ![image](https://user-images.githubusercontent.com/23140351/153867258-29c9752d-4bd2-413a-a524-ec290cf94539.png)

4.2. Copy the folder "src/MaskRCNN/TER-final" on your Google Drive in the path "MyDrive"

![image](https://user-images.githubusercontent.com/23140351/153867419-7f15039a-a9b8-4ba7-9b94-89c504d84e65.png)

4.3. Make sure that the folders “outputImagesWithMasksBlackAndWhite”, “outputImagesWithMaskColors”, “testAnnotations”, “testImages”, “trainImages” and “trainAnnotations” are empty (otherwise, empty them)

4.4. Put in the "testImages" folder the images on which you want to automatically create your annotations

4.5. Put in the "trainImages" folder the images on which you want to train the model

4.6. Put in the "trainAnnotations" folder the JSON annotation file corresponding to the images of the step 4.5
  - You can use the annotation file generated from the testing phase
  - You can use the annotation file generated from manually annotated images
  - You can use the annotation file generated from external sources
  
 4.7. Right click with the mouse on the file "finalNotebook_trainAndTest_COLAB.ipynb &arr open with &arr Google Colaboratory
 
 
![image](https://user-images.githubusercontent.com/23140351/153868066-3f8c52da-8223-4609-ae07-6e8659bb1666.png)

4.8. Click on Runtime &arr Change runtime type &arr Hardware accelerator &arr GPU &arr Save to enable the GPU acceleration

![image](https://user-images.githubusercontent.com/23140351/153868242-4e24aae2-4d13-430c-9108-1dc7b21e9aff.png)
![image](https://user-images.githubusercontent.com/23140351/153868259-cc6e73e9-9082-45a6-ac17-10950b0fd21d.png)

4.9. Click on "Runtime" &arr "Run all" to run the whole notebook

![image](https://user-images.githubusercontent.com/23140351/153868329-9c2b9d15-17c2-4b2b-83b0-5d5af59e4c4f.png)

4.10. Google Colab will you ask to mount your files (it is a mandatory step to perform)
- Click on "Connect to Google Drive"

![image](https://user-images.githubusercontent.com/23140351/153868440-3a9067af-fc55-4d0c-9941-d691f6733a81.png)

- Click on your account name

![image](https://user-images.githubusercontent.com/23140351/153868474-3e06f392-0e3e-4167-8242-bf670b94ca4d.png)

- Click on "Allow"

![image](https://user-images.githubusercontent.com/23140351/153868512-0300cf81-789c-4f0c-a170-caaa6b9ea060.png)


4.11. Go back to the Drive folder and download the "testAnnotations.json" file located in the folder "/MyDrive/TER-final/testAnnotations"

![image](https://user-images.githubusercontent.com/23140351/153868613-5f47b7dd-7309-4234-8f3c-a2907c5d8c66.png)

4.12. Open the UI installed locally and follow the instructions of steps 1 and 2 (import images and annotations &arr see the result)


![image](https://user-images.githubusercontent.com/23140351/153868709-649a72ea-fee6-4a04-b955-7914975adde0.png)

![image](https://user-images.githubusercontent.com/23140351/153868728-6f242bcf-ac69-4a22-a340-a2f5c9b1b0e1.png)


### 5. Testing in Google Colab

5.1. Download from the GitHub repository the branch “google-colab”
  - In a terminal window write “git clone google-colab https://github.com/alessiodimonte/TER.git”


5.2	Copy the folder “src/MaskRCNN/TER-final” on your Google Drive

5.3	Make sure that the folders “outputImagesWithMasksBlackAndWhite”, “outputImagesWithMaskColors”, “testAnnotations”, “testImages”, “trainImages”, “trainAnnotations” are empty (otherwise, empty them)

5.4	Put in the “testImages” folder the images on which you want to automatically create your annotations

5.5. Right click on the file "finalNotebook_test_COLAB.ipynb" &arr "Open with" &arr "Google Colaboratory"

5.6. Click on "Runtime" &arr "Change runtime type" &arr "Hardware accelerator" &arr "GPU" &arr "Save" to enable the GPU acceleration

5.7. Click on "Runtime" &arr; "Run all" to run the whole notebook

5.8. Google Colab will ask you to mount your files (it is a mandatory step to perform)
  - Click on "Connect to Google Drive"
  - Click on your account name
  - Click on "Allow"

5.9. Go back to the Drive folder and download the "testAnnotations.json" file located in the folder "/MyDrive/TER-final/testAnnotations". 


# More Information
For more information, please refer the Technical Report under the folder Technical Report in the master branch. (add link)
