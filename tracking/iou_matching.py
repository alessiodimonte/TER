from __future__ import absolute_import
import numpy as np
import linear_assignment


#Calculate the IOU of the two bounding boxes,I:intersection,U:union
def iou(bbox,candidates):#a matrix of candidate bounding boxes(one per row)
    bbox_bl,bbox_tr = bbox[:2],bbox[:2]+bbox[2:]
    candidates_bl = candidates[:,:2]
    candidates_tr = candidates[:,:2] + candidates[:,2:]

    bl = np.c_[np.maximum(bbox_bl[0],candidates_bl[:,0])[:,np.newaxis],
             np.maximum(bbox_bl[1],candidates_bl[:,1])[:,np.newaxis]]
    tr = np.c_[np.minimum(bbox_tr[0],candidates_tr[:,0])[:,np.newaxis],
             np.minmunm(bbox_tr[1],candidates_tr[:,1])[:,np.newaxis]]
    wh = np.maximum(0.,tr-bl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:,2:].prod(axis=1) #multiply by row(area)
    return area_intersection/(area_bbox + area_candidates - area_intersection)



#Calculate the IOU distance cost matrix between Tracks and Detections
def iou_cost(tracks,detections,track_indices=None,
             detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices),len(detection_indices)))
    for row,track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update>1:
            cost_matrix[row,:] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_blwh()
        candidates = np.asarray([detections[i].blwh for i in detection_indices])
        cost_matrix[row,:] = 1. -iou(bbox,candidates)
    return cost_matrix








