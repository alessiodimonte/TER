# To apply PNMdetector on videos

- Example : https://unice-my.sharepoint.com/:f:/g/personal/fanny_simoes_unice_fr/Euh7uD2NNLtGorAOkxkUt7MBJLu_UeyeCioDcdNmCqTYFg?e=e23PB8
- conda activate pyt36tf112cpu (or for gpu: conda activate pyt36tf112gpu)
- Use script : apply_pnmdetector_video.py

## To do previously

### 1) Installation of environment

- Follow the instruction in section 1 on github "train_pnmdetector":  Installation of Tensorflow 1 Object detection API

### 2) Modificate script: apply_pnmdetector_video.py

- line 15 : modificate link to videos

- line 37 + line 44 : modificate link to export and import images

- line 46 : modificate link to model saved (model trained then save). Our model called "export_model" is saved in the previous example (follow the link).

- line 48 : modificate link to labels (associated to model saved). Our labels called "label_map.pbtxt" are saved in the previous example (follow the link).

- line 54 : modificate link to export results of PNMdetector



