import glob
import os
import cv2
import datetime

#Link to import videos
link_dir_video = ".../*.MP4"

#Link to export images
link_img =".../images/"

#################################################
# Import videos
dir_video = glob.glob(link_dir_video)
# Split videos to images
for path_ani in dir_video:
    ind = dir_video.index(path_ani)
    vidcap = cv2.VideoCapture(path_ani)
    name_vid1 = os.path.basename(path_ani)
    name_vid = os.path.splitext(name_vid1)[0]
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
            cv2.imwrite(link_img + str(date_vid) + "_" + str(hour_vid) + "_" + str(name_vid) + "_frame%02d_vid%02d.JPG" % (counter, nbrvid), image)  # save frame as JPEG file
            counter += 1
        count += 1
