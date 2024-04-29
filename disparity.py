import cv2
import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser

class StereoSGBM:
    def __init__(self, minDisparity, blockSize, numDisparities):
        self.scale = 1 # pixel value scaling factor when loading input
        self.maxDisparity = numDisparities
        self.window = blockSize
        self.includeDiagonals = 0
        numPasses = 3
        self.downscale = 1
        self.threshold = False

        self.disparityS16 = None

    @staticmethod
    def create(minDisparity=0, blockSize=5, numDisparities=64):
        return StereoSGBM(minDisparity=minDisparity, blockSize=blockSize, numDisparities=numDisparities)

    def computeWithConfidence(self, cv_img_left, cv_img_right):
        # Load input into a vpi.Image and convert it to grayscale, 16bpp
        with vpi.Backend.CUDA:
            left = vpi.asimage(cv_img_left).convert(vpi.Format.Y16_ER, scale=self.scale)
            right = vpi.asimage(cv_img_right).convert(vpi.Format.Y16_ER, scale=self.scale)

        # Preprocess input
        confidenceU16 = None

        outWidth = (left.size[0] + self.downscale - 1) // self.downscale
        outHeight = (left.size[1] + self.downscale - 1) // self.downscale

        confidenceU16 = vpi.Image((outWidth, outHeight), vpi.Format.U16)

        # Estimate stereo disparity.
        with vpi.Backend.CUDA:
            self.disparityS16 = vpi.stereodisp(left, right, out_confmap=confidenceU16, mindisp=0, maxdisp=self.maxDisparity, includediagonals=self.includeDiagonals, window=self.window)
            # Disparities are in Q10.5 format, so to map it to float, it gets
            # divided by 32. Then the resulting disparity range, from 0 to
            # stereo.maxDisparity gets mapped to 0-255 for proper output.
            # Copy disparity values back to the CPU.
            disparity_signed = self.disparityS16.cpu()
            disparityU8 = self.disparityS16.convert(vpi.Format.U8, scale=255.0/(32*self.maxDisparity)).cpu()
            # # TODO: For some reason the disparity is inverted, so I had to invert it back
            # disparityU8 = 255 - disparityU8
            confidenceU8 = confidenceU16.convert(vpi.Format.U8, scale=255.0/65535).cpu()

        return disparityU8, confidenceU8

    def compute(self, cv_img_left, cv_img_right):
        disparityColor, confidenceU8 = self.computeWithConfidence(cv_img_left, cv_img_right)
        return disparityColor
