import json
import cv2
import os
import pandas as pd
from collections import Counter

with open('/workspace/fsimoes/detection/megadetector/results_megaBB/megaBB.json') as f:
  datajson = json.load(f)

#########################################
# Link of images with BB threshold 90% #
#########################################
file_BB90 = []
for p in datajson['images']:
    for q in p['detections']:
        if q['conf'] >= 0.9:
           file_BB90.append(p['file'])

file_BB90 = set(file_BB90) #unique link, when more 1 BB the link is repeated

multiple_label = ["biche_cerf","chamois_crave","humain_chien","renard_biche","renard_cerf","velo_humain","velo_humain_chien","voiture_humain"]

file_BB90_1label=[]
file_BB90_2label=[]
for x in file_BB90:
    if not any(value in x for value in multiple_label):
        file_BB90_1label.append(x)
    else:
        file_BB90_2label.append(x)

#########################################
#   new labels to BB with multi labels  #
#########################################
# import excel file with id of BB associated to their species
dfcsv = pd.read_csv('/workspace/fsimoes/detection/megadetector/rename_multilabels/BBmultilabels.csv',sep=";")

# import json file of dictionnary of file with multilabels
with open('/workspace/fsimoes/detection/megadetector/rename_multilabels/dict_2label.json') as g:
  dict_2label = json.load(g)

#rename label according to id of BB
for j in dict_2label['images']:
    localisation = dfcsv.loc[dfcsv['idname']==(j["id"])]
    j["label"] = list(localisation["species"])[0]

# check thanks to video
for l in dict_2label['images']:
    image = cv2.imread(l['filename'])
    height = image.shape[0]
    width = image.shape[1]
    x = int(l['bb'][0] * width)
    y = int(l['bb'][1] * height)
    h = int(l['bb'][3] * height)
    w = int(l['bb'][2] * width)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv2.putText(image, l['label'], (x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),6)
    cv2.imwrite(
        "/workspace/fsimoes/detection/megadetector/rename_multilabels/newlabels/" + "rename" + l['id'] + ".JPG", image)
    print("export rename",l['id'])

#remove id key
for k in dict_2label['images']:
    del k['id']
print("Number of BB for multi label: ",len(dict_2label['images']))

#remove label "vide"
dict_label = {"images":[]}
for m in dict_2label['images']:
    if m['label'] != 'vide':
        dict_label["images"].append(m)
print("Number of BB after remove label vide:",len(dict_label['images']))

#########################################
#       dic images with one BB          #
#########################################
cpteur=0
for a in datajson['images']:
    if any(value in a['file'] for value in file_BB90_1label):
        for b in a['detections']:
            if b['conf'] >= 0.9:
                dict2_label = {"filename": [], "bb": [], "label": []}
                cpteur += 1
                label = os.path.basename(a['file'])
                start = os.path.basename(a['file']).split("_")[:2]
                start = "_".join(start)+str("_")
                end = os.path.basename(a['file']).split("_")[-3:]
                end = str("_") + "_".join(end)
                label2 = label.replace(start,"")
                label3 = label2.replace(end,"")
                dict2_label["filename"] = (a['file'])
                dict2_label["bb"] = (b['bbox'])
                dict2_label["label"] = (label3)
                dict_label["images"].append(dict2_label)
                print("insert file",cpteur)

#print(dict_label['images'])
print("BB number:",len(dict_label['images']))

#BB number of species
BBspecies = []
for imgs in dict_label['images']:
    specie = imgs['label']
    BBspecies.append(specie)
print("Number of BB for each species", Counter(BBspecies))

print("Total number of images",len(file_BB90))
print("Total number of images with one label",len(file_BB90_1label))
print("Total number of images with multiple labels",len(file_BB90_2label))

#export json file
json.dump(dict_label, open("/workspace/fsimoes/detection/megadetector/sample_imgBB.json", 'w' ) )

#example of sample of video
lab=[]
for o in dict_label['images']:
    if o['label'] not in lab:
        name = os.path.basename(o['filename'])
        image = cv2.imread(o['filename'])
        height = image.shape[0]
        width = image.shape[1]
        x = int(o['bb'][0] * width)
        y = int(o['bb'][1] * height)
        h = int(o['bb'][3] * height)
        w = int(o['bb'][2] * width)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(image, o['label'], (x,y+h),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),6)
        lab.append(o['label'])
        cv2.imwrite(
            "/workspace/fsimoes/detection/megadetector/exvid_sample_imgBB/" + "img_" + o['label'] + ".JPG", image)
        print("export img",o['label'])











