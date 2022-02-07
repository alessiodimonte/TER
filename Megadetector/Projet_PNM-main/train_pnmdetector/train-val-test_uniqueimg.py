import json
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
import pandas as pd
import os

#####################################################
# TRAIN/VAL/TEST WITH DIFFERENT IMAGES IN EACH PART #
#####################################################

#### Import images / BB / labels
with open('.../sample_imgBB.json') as f:
  datajson = json.load(f)

#Json file to panda dataframe
for i in datajson:
    dfimgBB = pd.DataFrame(datajson[i])

species_keep = ['humain','chamois','chevreuil','biche','cerf','renard','blaireau','loup','lievre','chien','sanglier','velo','bouquetin']

# Link to save export images
train_img = ".../images/train/"
val_img = ".../images/val/"
test_img = ".../images/test/"

# Link to save csv file
train_csv = r'.../annotations/train.csv'
val_csv = r'.../annotations/val.csv'
test_csv = r'.../annotations/test.csv'

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
for path_train in train:
    name_img = os.path.basename(path_train)
    image = cv2.imread(path_train)
    cv2.imwrite(train_img + name_img, image)

# copy of val images
for path_val in val:
    name_img = os.path.basename(path_val)
    image = cv2.imread(path_val)
    cv2.imwrite(val_img + name_img, image)

# copy of test images
for path_test in test:
    name_img = os.path.basename(path_test)
    image = cv2.imread(path_test)
    cv2.imwrite(test_img + name_img, image)

################################################## Transform train/val/test to csv file
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
width = []
height = []

for i in dftrain['filename']:
    image = cv2.imread(i)
    width.append(image.shape[1])
    height.append(image.shape[0])
    cut = os.path.basename(i)
    filename_train.append(cut)
for j in dftrain['label']:
    label.append(j)
for k in dftrain['bb']:
    ymin.append(k[0])
    xmin.append(k[1])
    ymax.append(k[2])
    xmax.append(k[3])

dftrain2 = pd.DataFrame(list(zip(filename_train, width, height, label, ymin, xmin, ymax, xmax)),
               columns =['filename', 'width', "height", "class", "yminMD", "xminMD", "ymaxMD", "xmaxMD"])

#real coordinate of BB
dftrain2['xmin'] = dftrain2['yminMD'] * dftrain2['width']
dftrain2['ymin'] = dftrain2['xminMD'] * dftrain2['height']
dftrain2['xmax'] = dftrain2['xmin'] + (dftrain2['ymaxMD'] * dftrain2['width'])
dftrain2['ymax'] = dftrain2['ymin'] + (dftrain2['xmaxMD'] * dftrain2['height'])

print("Number of BB in train2", len(dftrain2))
dftrain2.to_csv(train_csv, index = False, header=True)

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
width = []
height = []

for i in dfval['filename']:
    image = cv2.imread(i)
    width.append(image.shape[1])
    height.append(image.shape[0])
    cut = os.path.basename(i)
    filename_val.append(cut)
for j in dfval['label']:
    label.append(j)
for k in dfval['bb']:
    ymin.append(k[0])
    xmin.append(k[1])
    ymax.append(k[2])
    xmax.append(k[3])

dfval2 = pd.DataFrame(list(zip(filename_val, width, height, label, ymin, xmin, ymax, xmax)),
               columns =['filename', 'width', "height", "class", "yminMD", "xminMD", "ymaxMD", "xmaxMD"])

dfval2['xmin'] = dfval2['yminMD'] * dftrain2['width']
dfval2['ymin'] = dfval2['xminMD'] * dftrain2['height']
dfval2['xmax'] = dfval2['xmin'] + (dfval2['ymaxMD'] * dftrain2['width'])
dfval2['ymax'] = dfval2['ymin'] + (dfval2['xmaxMD'] * dftrain2['height'])

print("Number of BB in val2", len(dfval2))
dfval2.to_csv(val_csv, index = False, header=True)

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
width = []
height = []

for i in dftest['filename']:
    image = cv2.imread(i)
    width.append(image.shape[1])
    height.append(image.shape[0])
    cut = os.path.basename(i)
    filename_test.append(cut)
for j in dftest['label']:
    label.append(j)
for k in dftest['bb']:
    ymin.append(k[0])
    xmin.append(k[1])
    ymax.append(k[2])
    xmax.append(k[3])

dftest2 = pd.DataFrame(list(zip(filename_test, width, height, label, ymin, xmin, ymax, xmax)),
               columns =['filename', 'width', "height", "class", "yminMD", "xminMD", "ymaxMD", "xmaxMD"])

dftest2['xmin'] = dftest2['yminMD'] * dftrain2['width']
dftest2['ymin'] = dftest2['xminMD'] * dftrain2['height']
dftest2['xmax'] = dftest2['xmin'] + (dftest2['ymaxMD'] * dftrain2['width'])
dftest2['ymax'] = dftest2['ymin'] + (dftest2['xmaxMD']  * dftrain2['height'])

print("Number of BB in test2", len(dftest2))
dftest2.to_csv(test_csv, index = False, header=True)

