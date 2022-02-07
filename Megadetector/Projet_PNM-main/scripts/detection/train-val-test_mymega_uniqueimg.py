import json
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
import pandas as pd
import os
import dopa

#####################################################
# TRAIN/VAL/TEST WITH DIFFERENT IMAGES IN EACH PART #
#####################################################

#### Import images / BB / labels
with open('/workspace/fsimoes/detection/megadetector/sample_imgBB.json') as f:
  datajson = json.load(f)

#Json file to panda dataframe
for i in datajson:
    dfimgBB = pd.DataFrame(datajson[i])

species_keep = ['humain','chamois','chevreuil','biche','cerf','renard','blaireau','loup','lievre','chien','sanglier','velo','bouquetin']

###################################################### Export images train/val/test
D = {}
for i in datajson:
    for j in datajson[i]:
        if j['label'] in species_keep:  # to keep some species
            b = D.get(j['filename'], [])
            b.append((j['label'], j['bb']))
            D[j['filename']] = b

# Train + Validation (trainfull) 90% and Testing 10%
(trainfull, test) = train_test_split(list(D.keys()), test_size=0.1, random_state=42)

# Train 80%  + Validation 10%
(train,val) = train_test_split(trainfull, test_size=0.11, random_state=42)

print("Number of unique images in train", len(train))
print("Number of unique images in val", len(val))
print("Number of unique images in test", len(test))

#copy of train images
def trainimg(path_train):
    name_img = path_train.replace("/workspace/fsimoes/images/","")
    image = cv2.imread(path_train)
    cv2.imwrite("/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_uniqueimg/images/train/" + name_img, image)

runs_train = [[path_train] for path_train in train]
dopa.parallelize(runs_train, trainimg)

# copy of val images
def valimg(path_val):
    name_img = path_val.replace("/workspace/fsimoes/images/","")
    image = cv2.imread(path_val)
    cv2.imwrite("/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_uniqueimg/images/val/" + name_img, image)

runs_val = [[path_val] for path_val in val]
dopa.parallelize(runs_val, valimg)

# copy of test images
def testimg(path_test):
    name_img = path_test.replace("/workspace/fsimoes/images/","")
    image = cv2.imread(path_test)
    cv2.imwrite("/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_uniqueimg/images/test/" + name_img, image)

runs_test = [[path_test] for path_test in test]
dopa.parallelize(runs_test, testimg)

################################################## Transfor train/val/test to csv file
# Train df to csv
dftrain = dfimgBB[dfimgBB['filename'].isin(train)]
dftrain = dftrain[dftrain['label'].isin(species_keep)]
print("Number of BB in train", len(dftrain))
print("dataframe training species", Counter(dftrain['label']))

filename_train = []
label = []
ymin = []
xmin = []
ymax = []
xmax = []

for i in dftrain['filename']:
    cut = os.path.basename(i)
    filename_train.append(cut)
for j in dftrain['label']:
    label.append(j)
for k in dftrain['bb']:
    ymin.append(k[0])
    xmin.append(k[1])
    ymax.append(k[2])
    xmax.append(k[3])

width = [1920] * len(dftrain)
height = [1080] * len(dftrain)

dftrain2 = pd.DataFrame(list(zip(filename_train, width, height, label, ymin, xmin, ymax, xmax)),
               columns =['filename', 'width', "height", "class", "yminMD", "xminMD", "ymaxMD", "xmaxMD"])

#real coordinate of BB
dftrain2['xmin'] = dftrain2['yminMD'] * 1920
dftrain2['ymin'] = dftrain2['xminMD'] * 1080
dftrain2['xmax'] = dftrain2['xmin'] + (dftrain2['ymaxMD'] * 1920)
dftrain2['ymax'] = dftrain2['ymin'] + (dftrain2['xmaxMD'] * 1080)

print("Number of BB in train2", len(dftrain2))
dftrain2.to_csv(r'/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_uniqueimg/annotations/train.csv', index = False, header=True)

# Validation df to csv
dfval = dfimgBB[dfimgBB['filename'].isin(val)]
dfval = dfval[dfval['label'].isin(species_keep)]
print("Number of BB in val ", len(dfval))
print("dataframe validation species", Counter(dfval['label']))

filename_val = []
label = []
ymin = []
xmin = []
ymax = []
xmax = []

for i in dfval['filename']:
    cut = os.path.basename(i)
    filename_val.append(cut)
for j in dfval['label']:
    label.append(j)
for k in dfval['bb']:
    ymin.append(k[0])
    xmin.append(k[1])
    ymax.append(k[2])
    xmax.append(k[3])

width = [1920] * len(dfval)
height = [1080] * len(dfval)

dfval2 = pd.DataFrame(list(zip(filename_val, width, height, label, ymin, xmin, ymax, xmax)),
               columns =['filename', 'width', "height", "class", "yminMD", "xminMD", "ymaxMD", "xmaxMD"])

dfval2['xmin'] = dfval2['yminMD'] * 1920
dfval2['ymin'] = dfval2['xminMD'] * 1080
dfval2['xmax'] = dfval2['xmin'] + (dfval2['ymaxMD'] * 1920)
dfval2['ymax'] = dfval2['ymin'] + (dfval2['xmaxMD'] * 1080)

print("Number of BB in val2", len(dfval2))
dfval2.to_csv(r'/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_uniqueimg/annotations/val.csv', index = False, header=True)

# Test df to csv
dftest = dfimgBB[dfimgBB['filename'].isin(test)]
dftest = dftest[dftest['label'].isin(species_keep)]
print("Number of BB in test ", len(dftest))
print("dataframe test species", Counter(dftest['label']))

filename_test = []
label = []
ymin = []
xmin = []
ymax = []
xmax = []

for i in dftest['filename']:
    cut = os.path.basename(i)
    filename_test.append(cut)
for j in dftest['label']:
    label.append(j)
for k in dftest['bb']:
    ymin.append(k[0])
    xmin.append(k[1])
    ymax.append(k[2])
    xmax.append(k[3])

width = [1920] * len(dftest)
height = [1080] * len(dftest)

dftest2 = pd.DataFrame(list(zip(filename_test, width, height, label, ymin, xmin, ymax, xmax)),
               columns =['filename', 'width', "height", "class", "yminMD", "xminMD", "ymaxMD", "xmaxMD"])

dftest2['xmin'] = dftest2['yminMD'] * 1920
dftest2['ymin'] = dftest2['xminMD'] * 1080
dftest2['xmax'] = dftest2['xmin'] + (dftest2['ymaxMD'] * 1920)
dftest2['ymax'] = dftest2['ymin'] + (dftest2['xmaxMD'] * 1080)

print("Number of BB in test2", len(dftest2))
dftest2.to_csv(r'/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_uniqueimg/annotations/test.csv', index = False, header=True)

