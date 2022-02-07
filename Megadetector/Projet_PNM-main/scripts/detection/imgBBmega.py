import json
import cv2
import os
import dopa

# this script allows to adjust threshold, here threshold = 90%
with open('/workspace/fsimoes/detection/megadetector/results_megaBB/megaBB.json') as f:
  datajson = json.load(f)

file_BB = []
nbr=0
def drawBBimg(p):
    global nbr
    nbr += 1
    print(nbr)
    image = cv2.imread(p['file'])
    height = image.shape[0]
    width = image.shape[1]
    for q in p['detections']:
        if q['conf'] >= 0.9:
            x = int(q['bbox'][0] * width)
            y = int(q['bbox'][1] * height)
            h = int(q['bbox'][3] * height)
            w = int(q['bbox'][2] * width)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
            file_BB.append(p['file'])
            cv2.imwrite("/workspace/fsimoes/detection/megadetector/imgBB_mega/" + os.path.basename(p['file']), image)
    if p['file'] not in file_BB:
         cv2.imwrite("/workspace/fsimoes/detection/megadetector/imgBB_mega/" + os.path.basename(p['file']), image)

runs = [[p] for p in datajson['images']]

dopa.parallelize(runs, drawBBimg)


