##**Compilie kalman_filter section**<br>Kalman filter uses the dynamic model of the system.Kalman filter is a recursive estimation, that is,
# the estimation of the current state can be calculated as long as the estimated value of the state at the last moment and the observed value
# of the current state are known, so there is no need to record the historical information of observation or estimation.
# source"https://en.wikipedia.org/wiki/Kalman_filter"

import numpy as np
import scipy.linalg

##Table for the 0.95 quantile of the chi-square distribution with N(1 to 9).
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


# Two stages of Kalman filtering:
# 1.predict the position of track at the next moment
# 2.update the predicted location based on Dection
class KalmanFilter(object): # A simple Kalman filter for tracking bounding boxes in image space with 8-dim state space(x,y,w,h,vx,vy,vw,vh).The previous four parameters plus their velocities
    def _init_(self):
        ndim, dt = 4, 1.
        # Create Kalman Filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        # According to the current state estimate,we can choose the motion and observation uncertainty
        # These weights control the amount of uncertainty in the model
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement  # measurement:bounding box(x,y,w,h) and it return mean vector and covariance matrix of the new track
        mean_vel = np.zeros_like(mean_pos)
        # Translates slice objects to concatenation along the first axis
        mean = np.r_[mean_pos, mean_vel]  # np._r stack two matrices in a row


        std=[
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance =np.diag(np.square(std))
        return mean,covariance



    def predict(self,mean,covariance):#mean:8 dim mean vector covariance:8*8 dim covariance matrix
        #kalman filter is predicted by the mean and covariance of the target at the previous time.
        std_pos=[
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel=[
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos,std_vel]))  #motion_cov is the covariance matrix of process noise
        mean = np.dot(self._motion_mat,mean)
        covariance = np.linalg.multi_dot((self._update_mat,covariance,self._motion_mat.T))+motion_cov
        return mean,covariance



    def project(self,mean,covariance):#Project state distribution to measurement space.
        std=[
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        #Covariance of noise during measurement
        innovation_cov = np.diag(np.square(std))
        #Map the mean vector to the detection space
        mean = np.dot(self._update_mat,mean)
        #Map the covariance matrix to the detection space
        covariance=np.linalg.multi_dot((self._update_mat,covariance,self._update_mat.T))
        return mean,covariance+innovation_cov



    def update(self,mean,covariance,measurement):#
        #Map the mean and covariance to the detection space,
        projected_mean,projected_cov = self.projet(mean,covariance)
        chol_factor,lower = scipy.linalg.cho_factor(#Compute the Cholesky decomposition of the matrix
            projected_cov,lower=True,check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve((chol_factor,lower),np.dot(covariance,self._update_mat.T).T,
                                             check_finite=False).T
        #Calculate the mean error of detection and track
        innovation = measurement - projected_mean
        #The updated mean vector
        new_mean = mean + np.dot(innovation,kalman_gain.T)
        #The updated covariance matrix
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain,projected_cov,kalman_gain.T))
        return new_mean,new_covariance



    #Compute gating distance between state distribution and measurements
    def gating_distance(self,mean,covariance,measurements,only_position=False):
        #Measurement :4*N dim matrix of N measurements,each in(x,y,w,h)
        #A suitable distance threshold can be obtained from `chi2inv95`. If`only_position` is False, the chi-square distribution has 4 degrees of freedom, otherwise 2.
        #If True, distance computation is done with respect to the bounding box center position only.
        mean,covariance = self.project(mean,covariance)
        if only_position:
            mean,covariance=mean[:2],covariance[:2,:2]
            measurements = measurements[:,:2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(cholesky_factor,d.T,
                                          lower=True,check_finite=False,overwrite_b=True)
        squares_maha=np.sum(z * z,axis=0) #Mahalanobis distance
        return squares_maha









