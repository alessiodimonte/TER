class TrackState:
  Tentative=1  #Newly created tracks before enough evidences have been collected
  Confirmed=2  #enough evidences have been collected
  Deleted=3   #Tracks are no longer alive and remove them from the set of alive tracks


class Track:
    def _init_(self, mean, cls_, covariance, track_id, n_init, max_age, featuer=None):
        self.mean = mean  # Mean vector of the initial state distribution
        self.cls_ = cls_
        self.covariance = covariance  # Covariance matrix of the initial state distribution
        self.track_id = track_id  # A unique track identifier
        self._n_init = n_init  # Number of consecutive detections before the track is confirmed
        self._max_age = max_age  # The maximum number of consecutive misses before the track state is set to "Deleted"
        self.hits = 1  # Total number of measurement updates.(Ps:The initial value is set to 1)
        self.age = 1  # Total number of frames since first occurance
        self.time_since_update = 0  # Total number of frames since last measurement update
        self.state = TrackState.Tentative  # The current track state,and the initial state we set is Tentative
        self.features = []  # A list: On each measurement update,the associated feature vector is added to this list
        if feature is not None:
            self.features.append(feature)



    def to_blwh(self):   #get the current position in bounding box format:(x,y):bottom left,(w,h):width,height
        res = self.mean[:4].copy()
        res[:2] -= res[2:]/2
        return res


    def to_bltr(self):  # get the current position in bounding box format(min(x,y),max(x,y))
        res = self.to_blwh()
        res[2:] = res[:2] + res[2:]  # top right(x,y)
        return res

    def predict(self,
                kf):  # using a Kalman filter prediction step to propagate the state distribution to the current time step
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1  # As time goes on ,the number increases by 1
        self.time_since_update += 1  # As time goes on,the number increases by 1

    def update(self, kf, detection):  # Perform Kalman filter measurement update step and update the feature cache
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xywh())
        self.features.append(detection.feature)
        self.cls_ = detection.cls_
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):  # Mark this track as missed
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):  # Returns True if this track is tentative
        return self.state == TrackState.Tentative

    def is_confirmed(self):  # Return True if this track is confirmed
        return self.state == TrackState.Confirmed

    def is_deleted(self):  # Return True if this track is dead and should be deleted
        return self.state == TrackState.Deleted