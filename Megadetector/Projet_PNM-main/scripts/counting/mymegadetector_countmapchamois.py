import json
import os
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from object_detection.utils import visualization_utils as vis_util

# apply mymegadetector
with open('/workspace/fsimoes/detection/megadetector/results_megaBB/megaBB.json') as f:
  datajson = json.load(f)

TEST_IMAGE_PATHS = []
NAME_IMAGE_PATHS = []
FRAME_IMAGE_PATHS = []
VID_IMAGE_PATHS = []
for i in datajson:
  for k in datajson['images']:
    name_video = os.path.basename(k['file'])
    if 'chamois' in name_video:
      if '2020-03-10' in name_video:
        if k['file'] not in TEST_IMAGE_PATHS:
          TEST_IMAGE_PATHS.append(k['file'])
          NAME_IMAGE_PATHS.append(os.path.basename(k['file']))
          FRAME_IMAGE_PATHS.append(os.path.basename(k['file']).split('_')[-2])
          VID_IMAGE_PATHS.append(os.path.basename(k['file']).split('_')[-1])

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_uniqueimg/training1/export_model/output_inference_graph_2000.pb/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/workspace/fsimoes/detection/mymegadetector/tensorflow1/tf1workspace/training_uniqueimg/annotations/label_map.pbtxt'

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load Frozen graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

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


nbr_bb = pd.DataFrame(columns=['detection_boxes','detection_scores','detection_classes','img'])
NAME_IMAGE_PATHS = pd.concat([pd.DataFrame(NAME_IMAGE_PATHS,columns=['img']),pd.DataFrame(FRAME_IMAGE_PATHS,columns=['frame']),pd.DataFrame(VID_IMAGE_PATHS,columns=['vid'])],axis=1)
NAME_IMAGE_PATHS = NAME_IMAGE_PATHS.sort_values(by='img')
print(NAME_IMAGE_PATHS)

for image_path in TEST_IMAGE_PATHS:
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
  #plt.imsave("/workspace/fsimoes/counting/img_BB_mymega" + os.path.basename(image_path) +".jpg", image_np)

  output_dict2 = {k: v.tolist() for k, v in output_dict.items()}
  df_output = pd.DataFrame(output_dict2)
  df_output['img'] = os.path.basename(image_path)
  df_output_good = df_output[df_output.detection_scores >= 0.6] #THRESHOLD TO FIXED
  nbr_bb = pd.concat([nbr_bb, df_output_good], ignore_index=True)

cpt_species = NAME_IMAGE_PATHS.merge(nbr_bb, on='img', how='left')
cpt_species = cpt_species.fillna(0)

imglist = cpt_species['img'].tolist()
df = pd.DataFrame(columns=["maille", "date", "vid"])
for image in imglist:
    maille_img = image.split('_')[0]
    date_img = image.split('_')[1]
    vid_img = image.split('_')[-1]
    df = df.append({
        "maille": maille_img,
        "date": date_img,
        "vid": vid_img
    }, ignore_index=True)
df=df.drop_duplicates()

grouped_df = cpt_species.groupby(['vid','frame','detection_classes'])
nbr_species_frame = grouped_df.size().reset_index(name='count_classes')
nbr_species_frame.loc[nbr_species_frame.detection_classes == 0, 'count_classes'] = 0
nbr_frame_by_classes = nbr_species_frame.groupby(['vid','detection_classes']).size().reset_index(name='nbr_frame_by_classes')
nbr_classes = nbr_species_frame.groupby(['vid','detection_classes'])[["count_classes"]].sum().reset_index()

final_number_species = nbr_classes.merge(nbr_frame_by_classes, on=['vid','detection_classes'], how='left')
final_number_species['nbr_species_vid'] = final_number_species['count_classes']/final_number_species['nbr_frame_by_classes']

nbr_species = 13
for i in range(1,nbr_species):
  final_number_species['detection_classes'].replace({0: "vide", i: category_index[i]['name']}, inplace=True)

# Filter by % of occurence of species
a = final_number_species.groupby(['vid'])[['nbr_frame_by_classes']].sum().reset_index()
a = a.rename(columns={"nbr_frame_by_classes":"nbr_total_frame"})
final_number_species2 = pd.merge(final_number_species,a)
final_number_species2['pct_classes_byframe'] = (final_number_species2['nbr_frame_by_classes']/final_number_species2['nbr_total_frame'])*100
final_number_species2 = final_number_species2.loc[final_number_species2['pct_classes_byframe'] >= 10]
final_number_species2 = pd.merge(final_number_species2,df)
print(final_number_species2)
final_number_species2.to_csv("/workspace/fsimoes/counting/final_number_species_chamois_2020-03-10.csv", index=False)
