import json
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2
import dopa
import pandas as pd
import os

#### Import images / BB / labels
with open('/workspace/fsimoes/detection/megadetector/sample_imgBB.json') as f:
  datajson = json.load(f)

pathimg = []
labelimg = []
bbimg = []

for i in datajson:
    for j in datajson[i]:
        if j['label'] not in ('pinson-des-arbres', 'crave', 'genette'): #to remove species
            pathimg.append(j['filename'])
            labelimg.append(j['label'])
            bbimg.append(j['bb'])

dfimg = pd.DataFrame(list(zip(pathimg, bbimg)), columns = ['filename', 'bb'])
dflabel = pd.DataFrame(labelimg, columns = ['label'])

(trainfullX, testX, trainfullY, testY) = train_test_split(dfimg, dflabel, test_size=0.2, random_state=42)

print("training length", len(trainfullX), len(trainfullY))
print("training", Counter(trainfullY['label']))

print("testing length", len(testX), len(testY))
print("testing", Counter(testY['label']))

#copy of training images
def trainimg(path_train):
    name_img = path_train.replace("/workspace/fsimoes/images/","")
    image = cv2.imread(path_train)
    cv2.imwrite("/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_demo/images/train/" + name_img, image)

runs_train = [[path_train] for path_train in trainfullX['filename']]
dopa.parallelize(runs_train, trainimg)

# copy of test images
def testimg(path_test):
    name_img = path_test.replace("/workspace/fsimoes/images/","")
    image = cv2.imread(path_test)
    cv2.imwrite("/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_demo/images/test/" + name_img, image)

runs_test = [[path_test] for path_test in testX['filename']]
dopa.parallelize(runs_test, testimg)

#################################
# Train df to csv
dftrain = pd.concat([trainfullX, trainfullY], axis=1)
print("dataframe training length", len(dftrain))

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

print("dataframe training2 length", len(dftrain2))
dftrain2.to_csv(r'/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_demo/annotations/train.csv', index = False, header=True)

# Test df to csv
dftest = pd.concat([testX, testY], axis=1)
print("dataframe testing length", len(dftest))

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

print("dataframe test2 length", len(dftest2))
dftest2.to_csv(r'/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_demo/annotations/test.csv', index = False, header=True)
