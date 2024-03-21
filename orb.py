
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
        # Convert from CPU to GPU
        # TODO: convert the keypoints
        return keypoints

    def detectAndComputeAsync(self, cv_stereo_img, mask=None):
        with vpi.Backend.CUDA:
            src = vpi.asimage(cv_stereo_img).convert(vpi.Format.U8)
            pyr = src.gaussian_pyramid(3)
            corners, descriptors = pyr.orb(intensity_threshold=142, max_features_per_level=88, max_pyr_levels=3)

        out = src.convert(vpi.Format.BGR8, backend=vpi.Backend.CUDA)

        # Draw the keypoints in the output image
        if corners.size > 0:
            distances = []
            with descriptors.rlock_cpu() as descriptors_data:
                first_desc = descriptors_data[0][0]
                for i in range(descriptors.size):
                    curr_desc = descriptors_data[i][0]
                    hamm_dist = sum([bin(c ^ f).count('1') for c, f in zip(curr_desc, first_desc)])
                    distances.append(hamm_dist)
            max_dist = max(distances)
            max_dist = max(max_dist, 1)
            cmap = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)
            cmap_idx = lambda i: int(round((distances[i] / max_dist) * 255))

            with out.lock_cpu() as out_data, corners.rlock_cpu() as corners_data:
                for i in range(corners.size):
                    color = tuple([int(x) for x in cmap[cmap_idx(i), 0]])
                    kpt = tuple(corners_data[i].astype(np.int16))
                    cv2.circle(out_data, kpt, 3, color, -1)

        return corners.cpu(), descriptors.cpu()
