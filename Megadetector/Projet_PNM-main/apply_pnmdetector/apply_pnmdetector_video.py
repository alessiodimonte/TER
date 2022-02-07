import glob
import os
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import datetime
from matplotlib import pyplot as plt
from object_detection.utils import visualization_utils as vis_util
import re

# Import videos
dir_video = glob.glob(".../maille*/*/*.MP4")
# Split videos to images
for path_ani in dir_video:
    ind = dir_video.index(path_ani)
    vidcap = cv2.VideoCapture(path_ani)
    name_vid1 = os.path.basename(path_ani)
    name_vid = os.path.splitext(name_vid1)[0]
    maille_vid = path_ani.rsplit('/', 3)[1]
    hourdate_vid = datetime.datetime.fromtimestamp(os.path.getmtime(path_ani))
    date_vid = str(hourdate_vid.year) + "-" + str('{:02d}'.format(hourdate_vid.month)) + "-" + str('{:02d}'.format(hourdate_vid.day))
    hour_vid = str('{:02d}'.format(hourdate_vid.hour)) + "h" + str('{:02d}'.format(hourdate_vid.minute)) + "min"
    counter = 1
    count = 0
    nbrvid = ind + 1
    success = True
    while success:
        success, image = vidcap.read()
        if image is None:
            print(
                'Read a new frame: False, all images are already transferred')  # no images to read in the video, all images are already read
        elif count % 30 == 0:
            print('Read a new frame:', success)
            cv2.imwrite(".../images/" + str(
                maille_vid) + "_" + str(date_vid) + "_" + str(hour_vid) + "_" + str(name_vid) + "_frame%02d_vid%02d.JPG" % (counter, nbrvid), image)  # save frame as JPEG file
            counter += 1
        count += 1

# import images
DIR_IMG = glob.glob(".../images/*.JPG")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '.../export_model/output_inference_graph_2000.pb/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '.../label_map.pbtxt'

# Threshold to fixed
th = 0.6

# Link to export results
link_results = ".../resultats_especes_videos.csv"

#################################################################################################################################@

###################
# Apply our Model
###################

# Create list of images with their names
NAME_IMAGE_PATHS = []
FRAME_IMAGE_PATHS = []
VID_IMAGE_PATHS = []
for img in DIR_IMG:
    NAME_IMAGE_PATHS.append(os.path.basename(img))
    FRAME_IMAGE_PATHS.append(os.path.basename(img).split('_')[-2])
    VID_IMAGE_PATHS.append(os.path.basename(img).split('_')[-1])
#print(TEST_IMAGE_PATHS)
#print(NAME_IMAGE_PATHS)

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#print(category_index)

# Load Frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Image to numpy array
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Run mymegadetector on images
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
           'detection_boxes', 'detection_scores', 'detection_classes'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# Only keep detection_boxes, detection_scores and detection_classes
nbr_bb = pd.DataFrame(columns=['detection_boxes','detection_scores','detection_classes','img'])
NAME_IMAGE_PATHS = pd.concat([pd.DataFrame(NAME_IMAGE_PATHS,columns=['img']),pd.DataFrame(FRAME_IMAGE_PATHS,columns=['frame']),pd.DataFrame(VID_IMAGE_PATHS,columns=['vid'])],axis=1)
NAME_IMAGE_PATHS = NAME_IMAGE_PATHS.sort_values(by='img')

for image_path in DIR_IMG:
  print(image_path)
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection
  output_dict = run_inference_for_single_image(image_np, detection_graph)

  # Visualization of the results of a detection.
  #vis_util.visualize_boxes_and_labels_on_image_array(
  #  image_np,
  #  output_dict['detection_boxes'],
  #  output_dict['detection_classes'],
  #  output_dict['detection_scores'],
  #  category_index,
  #  instance_masks=output_dict.get('detection_masks'),
  #  use_normalized_coordinates=True,
  #  line_thickness=8,
  #  min_score_thresh=0.6) #THRESHOLD TO FIXED
  #plt.imshow(image_np)
  #plt.imsave("/workspace/fsimoes/counting/img_BB_mymega/" + os.path.basename(image_path), image_np)

  output_dict2 = {k: v.tolist() for k, v in output_dict.items()}
  df_output = pd.DataFrame(output_dict2)
  df_output['img'] = os.path.basename(image_path)
  df_output_good = df_output[df_output.detection_scores >= th] #THRESHOLD TO FIXED
  nbr_bb = pd.concat([nbr_bb, df_output_good], ignore_index=True)

###################
# Count species
###################

cpt_species = NAME_IMAGE_PATHS.merge(nbr_bb, on='img', how='left')
cpt_species = cpt_species.fillna(0)

#name img
imglist = cpt_species['img'].tolist()
df = pd.DataFrame(columns=["maille", "date", "heure", "vid", "nom"])
for image in imglist:
    name_img = re.sub('frame\d+', 'frame*', image)
    maille_img = image.split('_')[0]
    date_img = image.split('_')[1]
    hour_img = image.split('_')[2]
    vid_img = image.split('_')[-1]
    df = df.append({
        "maille": maille_img,
        "date": date_img,
        "heure": hour_img,
        "vid": vid_img,
        "nom": name_img
    }, ignore_index=True)
df=df.drop_duplicates()

#Number detection_classes(species) by frame
grouped_df = cpt_species.groupby(['vid','frame','detection_classes'])
nbr_species_frame = grouped_df.size().reset_index(name='count_classes')
nbr_species_frame.loc[nbr_species_frame.detection_classes == 0, 'count_classes'] = 0
#Number frame by detection_classes
nbr_frame_by_classes = nbr_species_frame.groupby(['vid','detection_classes']).size().reset_index(name='nbr_frame_by_classes')
#Number total of detection_classes by video
nbr_classes = nbr_species_frame.groupby(['vid','detection_classes'])[["count_classes"]].sum().reset_index()
#Number of species by video
final_number_species = nbr_classes.merge(nbr_frame_by_classes, on=['vid','detection_classes'], how='left')
final_number_species['nbr_species_vid'] = final_number_species['count_classes']/final_number_species['nbr_frame_by_classes']

#Replace detection_classe by real name
nbr_species = len(category_index)
for i in range(1,nbr_species):
  final_number_species['detection_classes'].replace({0: "vide", i: category_index[i]['name']}, inplace=True)

# Filter by % of occurence of species

# nbr total frame by video
a = final_number_species.groupby(['vid'])[['nbr_frame_by_classes']].sum().reset_index()
a = a.rename(columns={"nbr_frame_by_classes":"nbr_total_frame"})
final_number_species2 = pd.merge(final_number_species,a)

# % of detection classe on video
final_number_species2['pct_classes_byframe'] = (final_number_species2['nbr_frame_by_classes']/final_number_species2['nbr_total_frame'])*100
final_number_species2 = final_number_species2.loc[final_number_species2['pct_classes_byframe'] >= 10]
final_df = pd.merge(final_number_species2,df)

# Remove element "vide" but not video vide !
final_df = final_df.drop(final_df[(final_df['detection_classes'] == 'vide') & (final_df['pct_classes_byframe'] < 100)].index) # If 100% frame are "vide" so video is "vide"
# select good columns and rename
final_df = final_df[["nom","maille", "date", "heure", "detection_classes", 'nbr_species_vid']]
final_df = final_df.rename(columns={"detection_classes":"espece","nbr_species_vid":"nbr_espece"})

###############
#export results
###############
final_df.to_csv(link_results, index=False)
