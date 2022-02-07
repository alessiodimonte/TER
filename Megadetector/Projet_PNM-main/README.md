# PNMdetector V1.0

## To train model: 

- If you use new images, go to file Projet_PNM/annotated_pnmdetector

1st: Split your videos into images 

2nd: Annotated your images 

- Then whatever you use new images or not go to file Projet_PNM/train_pnmdetector

The current model is based on MegaDetector model V4.1. (https://github.com/microsoft/CameraTraps/blob/master/megadetector.md), based on Faster-RCNN with an InceptionResNetv2 base network and a Softmax layer. 13 species are retained: humain, chamois, chevreuil, biche, cerf, renard, blaireau, loup, lievre, chien, sanglier, velo and bouquetin. The model is trained on 53,203 bounding boxes correspond to 38,158 images. Images are different in train-validation-test part.

## To apply model: 

- Go to file Projet_PNM/apply_pnmdetector

At the moment, it is only applicable on files which contains video. The model applied is the last model trained in the last section.

Method to count species : 
To obtain good detection for all species, a threshold of 0.6 is fixed. When we tried 0.5, lot of false detections were retained whereas when it was fixed to 0.7 few true detections were retained. Therefore, the threshold 0.6 seems to be the best option. So, if detection_scores are < 0.6, we consider images are empty (detection_classes = 0 = “vide”). Firstly, we calculate number of species detected by video and the number of frames where species were detected. Secondly, we count total number of frames on videos. If a specie represents less than 10\% of image, then it is a false detection, we remove this specie. Concerning videos, if 100\% of frames are detection_classes=“vide” then videos are named “vide”. We have create a CSV file summary, listing every species’ detections on videos, for each video we classify that way : nom, maille, date, heure, espece, nombre d’espèce.

Globally, PNMdetector seems to work fine for detection but for classification, results are not very good, it seems to work very well only for big species like humain (around detection_scores > 90\%), chamois, chevreuil and renard. 

To realise map of repartition of species you can use ods file with localisation of camera by maille: videos_mercantour_23_06_2020/coordonnees_pieges-photos_2020_4.ods

## Improvement axes: 

### To train model: 

-	Change configuration of model: pnmdetector.config (replace softmax, batch size, steps, metric_set, data augmentation, size images, optimizer).

-	Change classes (add news classes or create classes which gather multiple species).

-	Train with more images.

-	Images train-validation-test from different video or maille.

### To apply model: 

-	Possibility to apply PNMdetector on files which contain only images.

-	Method of counting (when two species are not present at same moment, indicate false detection, change threshold fixed).
