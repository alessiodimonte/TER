from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
import kalman_filter

INFTY_COST = 1e+5

#Gated cosine distance cost
def min_cost_matching(
        distance_metric,max_distance,tracks,detections,track_indices=None,
        detection_indices=None):#Solve linear assignment problem.
    if track_indices is None:
        track_indices = np.arange(len(tracks))
        #tracks:A list of predicted tracks at the current time step
        #track_indices:List of track indices that maps rows in `cost_matrix` to tracks in `tracks`
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
        #detections:A list of detections at the current time step
        #detection_indices:List of detection indices that maps columns in `cost_matrix` to detections in `detections`
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [],track_indices,detection_indices

    # Calculate the cost matrix
    cost_matrix = distance_metric(tracks,detections,track_indices,detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    #Hungarian algorithm is executed to get the index pair assigned successfully. The row index is tracks index,
    # and the column index is Detections index
    row_indices,col_indices = linear_assignment(cost_matrix)
    matches,unmatched_tracks,unmatched_detections = [],[],[]

    #Find the unmatched Detections
    for col,detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    #Find the unmatched Tracks
    for row,track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    #To traverse the matched index pairs
    for row,col in zip(row_indices,col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        #If the corresponding cost is greater than the threshold max_distance, the match is considered unsuccessful
        if cost_matrix[row,col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx,detection_idx))
    return matches,unmatched_tracks,unmatched_detections



def matching_cascade(distance_metric,max_distance,cascade_depth,tracks,detections,
                     track_indices=None,detection_indices=None):
    #Assign track_indices and DETECtion_indices lists
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    unmatched_detections = detection_indices
    matches = []
    # match tracks for each level from small to large
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break # If not detections, exit the loop
        #Select tracks by age
        track_indices_1 = [
            k for k in track_indices
            if tracks[k].time_science_update == 1+level
        ]
        if len(track_indices_1) == 0:
            continue #if the current level has no track, continue
        #Call the min_cost_matching function to match
        matches_1,_,unmatched_detections= \
            min_cost_matching(distance_metric,max_distance,tracks,detections,
                              track_indices_1,unmatched_detections)
        matches += matches_1
    unmatched_tracks = list(set(track_indices)-set(k for k,_ in matches))
    return matches,unmatched_tracks,unmatched_detections
#Gated cost matrix: limit the cost matrix by calculating the distance between kalman filter state distribution and measured values.
#The distance in cost matrix is the appearance similarity between Track and detection.
#If a trajectory is to match two detection with very similar appearance features, it is easy to make mistakes.
#Let the two detection algorithms calculate the Mahalanobis distance from the trajectory and limit it with a threshold gating_threshold.
#The detection with the farther Mahalanobis distance can be separated, thus reducing false matches.



def gate_cost_matrix(
        kf,cost_matrix,tracks,detections,track_indices,detection_indices,
        gated_cost=INFTY_COST,only_position=False):
    gating_dim= 2 if only_position else 4
    gating_threshold = kalman_fiflter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xywh() for i in detection_indices])
    for row,track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean,track.covariance,measurements,only_position)
        cost_matrix[row,gating_distance > gating_threshold] = gated_cost
    return cost_matrix

