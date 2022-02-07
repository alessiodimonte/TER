import pandas as pd
import numpy as np
import glob
import os
import cv2
from natsort import natsorted
import time
import dopa

# Import excel file (name of species)
path_ods = "/workspace/fsimoes/videos_mercantour_23_06_2020/releve_mars_piege-photo.ods"  # change

# Import videos
dir_video = glob.glob(
    "/workspace/fsimoes/videos_mercantour_23_06_2020/maille*/*/*.MP4")  # change
dir_video = natsorted(dir_video)

# File to save images
link_img = "/workspace/fsimoes/images/"  # change

###########################################################################################################
# import only videos with names associated (no empty video)
maille_video = []
date_video = []
# Recover names of species/labels of videos
label_vid = []
nbr_animal = []
dir_animal = []
dir_noanimal =[]

for path in dir_video:
    nm_video = os.path.basename(path)
    nm_video1 = os.path.splitext(nm_video)[0]  # extract name of video
    ml_video = path.rsplit('/', 3)[1]  # extract name of maille
    #d_video = path.rsplit('/', 2)[1]  # date of plan video
    d_video = time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(path))) # real date of video
    df = pd.read_excel(path_ods, ml_video, header=3, engine="odf", keep_default_na=True,
                       na_values=' ')  # open excel file with name of "onglet" (=name of maille), ' ' considered as NA
    df = df[['numero_video', 'espece', 'nombre_individus']]  # keep only column numero video, espece, nombre individus
    df['espece'].fillna(value='vide', inplace=True)  # if NA in column "espece" then equal "vide"
    df['espece'] = np.where(df.espece.str.contains("lievre-gris"), "lievre",
                            df['espece'])  # rename lievre-gris by lievre
    df['espece'] = np.where(df.espece.str.contains("lievre-europe"), "lievre",
                            df['espece'])  # rename lievre-europe by lievre
    df['nombre_individus'].fillna(value=1, inplace=True)  # if NA in column "nombre individu" then equal 1
    df['nombre_individus'] = np.where(df.espece.str.contains("vide"), "0",
                                      df['nombre_individus'])  # if espece = "vide" then nombre_individus = 0
    df['nombre_individus'] = df['nombre_individus'].astype(float)
    df['nombre_individus'] = df['nombre_individus'].astype(int)

    if nm_video1 in list(df["numero_video"]):  # if the name of video equal to column "numero video" then equal column "espece"
        nm1 = df.loc[df["numero_video"] == nm_video1]
        if list(nm1["espece"]) != ["vide"]:
            species = list(nm1["espece"])
            nbr_animals = list(nm1["nombre_individus"])
            label_vid.append(species[0])
            nbr_animal.append(nbr_animals[0])
            maille_video.append(ml_video)
            date_video.append(d_video)
            dir_animal.append(path)
        else:
            dir_noanimal.append(path)
    else:
        dir_noanimal.append(path)


# Split videos into images with number of videos
def cutvid(path_ani):
    ind = dir_animal.index(path_ani)
    vidcap = cv2.VideoCapture(path_ani)
    name_vid = label_vid[ind]
    maille_vid = maille_video[ind]
    date_vid = date_video[ind]
    nbr_animal_vid = nbr_animal[ind]
    counter = 1
    count = 0
    nbrvid = ind + 1
    success = True
    while success:
        success, image = vidcap.read()
        if image is None:
            print('Read a new frame: False, all images are already transferred') # no images to read in the video, all images are already read
        elif count % 30 == 0:
            print('Read a new frame:', success)
            cv2.imwrite(link_img + str(
                maille_vid) + "_" + str(date_vid) + "_" + str(name_vid) + "_x" + str(
                nbr_animal_vid) + "_frame%02d_vid%02d.JPG" % (counter, nbrvid), image)  # save frame as JPEG file
            counter += 1
        count += 1

runs = [[path_ani] for path_ani in dir_animal]

dopa.parallelize(runs, cutvid)
print(runs)
