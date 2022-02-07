import pandas as pd
import numpy as np
from collections import Counter

# run to local

# Import excel file (name of species)
path_ods = "/Users/simoesfanny/Documents/Projet_Mercantour/data/extraits_videos_23_06_2020/releve_mars_piege-photo.ods" #change

################################################################ Which species and how many videos or images of species?
dataf = pd.read_excel(path_ods,sheet_name=None,header=3,engine="odf",keep_default_na=True,na_values=' ')
key = dataf.keys()

species = []
for i in key:
  df = dataf[i]
  df = df[['numero_video','espece']]
  df['espece'].fillna(value='vide', inplace = True) # if NA in column "espece" then equal "vide"
  df['espece'] = np.where(df.espece.str.contains("lievre-gris"), "lievre",
                          df['espece'])  # rename lievre-gris by lievre
  df['espece'] = np.where(df.espece.str.contains("lievre-europe"), "lievre",
                          df['espece'])  # rename lievre-europe by lievre
  specie = list(df['espece'])
  species.append(specie)

species = sum(species, [])
print("Total images and videos number:",len(species)) #Total number of videos or images reference in ods file (careful some videos/images are not referenced in ods file due to empty)
print("Different species in images/videos:",sorted(set(species))) #labels/species in our data
print("Different species in images/videos:",(Counter(species)))#number of videos/images of each species (careful number "vide" is false)

################################################################ How many images ?
sheet_name = ["maille00","maille72","maille119","maille117","maille69","maille66","maille23"]
print("Files number with images :",len(sheet_name))

species_img=[]
for i in range(0,len(sheet_name)): #according to number of videos
  sheet_name_img = sheet_name[i] #recover name of maille
  df = pd.read_excel(path_ods,sheet_name_img,header=3,engine="odf",keep_default_na=True,na_values=' ') #open excel file with name of "onglet" (=name of maille), ' ' considered as NA
  df = df[['numero_video','espece','nombre_individus']] # keep only column numero video, espece, nombre individus
  df['espece'].fillna(value='vide', inplace = True) # if NA in column "espece" then equal "vide"
  df['espece'] = np.where(df.espece.str.contains("lievre-gris"), "lievre",
                          df['espece'])  # rename lievre-gris by lievre
  df['espece'] = np.where(df.espece.str.contains("lievre-europe"), "lievre",
                          df['espece'])  # rename lievre-europe by lievre
  a = list(df['espece'])
  species_img.append(a)

species_img2 = sum(species_img, [])
print("Images number:",len(species_img2)) #Total number of videos or images reference in ods file (careful some videos/images are not referenced in ods file due to empty)
print("Different species in images:",sorted(set(species_img2))) #labels/species in our data
print("Images number of each species:",(Counter(species_img2)))#number of videos/images of each species (careful number "vide" is false")

################################################################ How many videos ?
sheet_name = ["maille118","maille116","maille104","maille103","maille102","maille101","maille100","maille70","maille67","maille56","maille54","maille52","maille50","maille40","maille36","maille22","maille08","maille06","maille88","maille87","maille86","maille85","maille84","maille73","maille71","maille68","maille57","maille55","maille53","maille51","maille41","maille38","maille24","maille21","maille07"]
print("Files number with videos :",len(sheet_name))

species_vid=[]
for i in range(0,len(sheet_name)): #according to number of videos
  sheet_name_vid = sheet_name[i] #recover name of maille
  df = pd.read_excel(path_ods,sheet_name_vid,header=3,engine="odf",keep_default_na=True,na_values=' ') #open excel file with name of "onglet" (=name of maille), ' ' considered as NA
  df = df[['numero_video','espece','nombre_individus']] # keep only column numero video, espece, nombre individus
  df['espece'].fillna(value='vide', inplace = True) # if NA in column "espece" then equal "vide"
  df['espece'] = np.where(df.espece.str.contains("lievre-gris"), "lievre",
                          df['espece'])  # rename lievre-gris by lievre
  df['espece'] = np.where(df.espece.str.contains("lievre-europe"), "lievre",
                          df['espece'])  # rename lievre-europe by lievre
  a = list(df['espece'])
  species_vid.append(a)

species_vid2 = sum(species_vid, [])
print("Videos number:",len(species_vid2)) #Total number of videos reference in ods file (careful some videos are not referenced in ods file due to empty)
print("Different species in videos:",sorted(set(species_vid2))) #labels/species in our videos
print("Videos number of each species:",(Counter(species_vid2)))#number of videos of each species (careful number "vide" is false")




