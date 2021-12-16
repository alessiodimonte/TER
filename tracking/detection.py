import numpy as np



class Detection(object): # This class represents a bounding box detection in a single image
    def __init__(self,blwh,confidence,feature):
        self.blwh = np.asarray(blwh,dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature,dtype=np.float32)



    def to_bltr(self):
        res = self.blwh.copy()
        res[2:] += res[:2]
        return res



    def to_xywh(self):
        res = self.blwh.copy()
        res[:2] += res[2:]/2
        return res
