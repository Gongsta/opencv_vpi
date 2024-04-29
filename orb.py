
import numpy as np
import vpi
import cv2

class ORB:
    # Visitor Pattern
    def __init__(self, nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold):
        self.nfeatures = nfeatures
        self.scaleFactor = scaleFactor
        self.nlevels = nlevels
        self.edgeThreshold = edgeThreshold
        self.firstLevel = firstLevel
        self.WTA_K = WTA_K
        self.scoreType = scoreType
        self.patchSize = patchSize

    @staticmethod
    def create(nfeatures = 500, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstLevel = 0, WTA_K = 2, scoreType = None, patchSize = 31, fastThreshold = 2):
        return ORB(nfeatures=nfeatures, scaleFactor=scaleFactor, nlevels=nlevels, edgeThreshold=edgeThreshold, firstLevel=firstLevel, WTA_K=WTA_K, scoreType=scoreType, patchSize=patchSize, fastThreshold=fastThreshold)

    def convert(self, keypoints):
        # Convert from GPU to CPU
        # TODO: convert the keypoints
        return keypoints

    # TODO: Add detect and compute
    # https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html

    def detectAndComputeAsync(self, cv_stereo_img, mask=None):
        return self.detectAndCompute(cv_stereo_img, mask)

    def detectAndCompute(self, cv_stereo_img, mask=None):
        with vpi.Backend.CUDA:
            src = vpi.asimage(cv_stereo_img).convert(vpi.Format.U8)
            pyr = src.gaussian_pyramid(self.nlevels)
            corners, descriptors = pyr.orb(intensity_threshold=self.edgeThreshold, max_features_per_level=int(self.nfeatures /self.nlevels), max_pyr_levels=self.nlevels)

        kpts_cpu = []
        with corners.rlock_cpu() as corners_data:
            for i in range(corners.size):
                x,y = corners_data[i].astype(np.int16)
                kpts_cpu.append(cv2.KeyPoint(x,y, 3))

        with descriptors.rlock_cpu() as descriptors_data:
            desc_cpu = np.array([record['data'] for record in descriptors_data])

        return kpts_cpu, desc_cpu
