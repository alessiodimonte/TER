import json
import cv2
import os

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

file_BB90 = set(file_BB90) #unique link, when more 1 BB in same img the link is repeated

multiple_label = ["biche_cerf","chamois_crave","humain_chien","renard_biche","renard_cerf","velo_humain","velo_humain_chien","voiture_humain"]

file_BB90_1label=[]
file_BB90_2label=[]
for x in file_BB90:
    if not any(value in x for value in multiple_label):
        file_BB90_1label.append(x)
    else:
        file_BB90_2label.append(x)

print("Number of images with multiple BB :",len(file_BB90_2label))

#########################################
#  dictionary images with multiple BB   #
#########################################
idname=0 #attribute id to each BB
dict_2label = {"images":[]}
for a in datajson['images']:
    if any(value in a['file'] for value in file_BB90_2label):
        for b in a['detections']:
            if b['conf'] >= 0.9:
                dict2_2label = {"filename": [], "bb": [], "label": [], "id":[]}
                idname +=1
                dict2_2label["id"] = ("id"+ str(idname))
                label = os.path.basename(a['file'])
                start = os.path.basename(a['file']).split("_")[:2]
                start = "_".join(start)+str("_")
                end = os.path.basename(a['file']).split("_")[-3:]
                end = str("_") + "_".join(end)
                label2 = label.replace(start,"")
                label3 = label2.replace(end,"")
                dict2_2label["filename"] = (a['file'])
                dict2_2label["bb"] = (b['bbox'])
                dict2_2label["label"] = (label3)
                dict_2label["images"].append(dict2_2label)
                print("add to dic", idname)


#print(dict_2label)
#print(dict_2label['images'])
print("BB:",idname)

# export to json file
json.dump(dict_2label, open("/workspace/fsimoes/detection/megadetector/rename_multilabels/dict_2label.json", 'w' ) )

#export images with BB and id associated
cpt = 0
for i in dict_2label['images']:
    cpt += 1
    image = cv2.imread(i['filename'])
    height = image.shape[0]
    width = image.shape[1]
    x = int(i['bb'][0] * width)
    y = int(i['bb'][1] * height)
    h = int(i['bb'][3] * height)
    w = int(i['bb'][2] * width)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    cv2.putText(image, i['id'], (1750, 1060), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 6)
    cv2.imwrite(
        "/workspace/fsimoes/detection/megadetector/rename_multilabels/idBB/" + i['id'] + ".JPG", image)
    print("export",cpt)

