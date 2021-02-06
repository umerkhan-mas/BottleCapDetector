import cv2 as cv
import numpy as np
import os, glob
from sklearn import svm, neighbors, tree , ensemble, naive_bayes 
from BottleCapDetector.Helpers.Helper import CropBoundingBox, ClusterImage
import pickle

winSize = (32,32)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv.HOGDescriptor(winSize,
                        blockSize,
                        blockStride,
                        cellSize,
                        nbins,
                        derivAperture,
                        winSigma,
                        histogramNormType,
                        L2HysThreshold,
                        gammaCorrection,
                        nlevels)

winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

class Hog_Predictor:
    def __init__(self, image, model_directory='./BottleCapDetector/Models'):
        self.__image__ = image
        self.__LoadModel__(model_directory)

    def __LoadModel__(self, model_directory):
        filename = 'svm_poly' + '.sav'
        file_path = os.path.join(model_directory, filename)
        self.__model__ = pickle.load(open(file_path, 'rb'))

    def PredictContours(self, contours):
        predictions = {}
        # Initialize all labels as empty lists
        for label in self.__model__.classes_:
            predictions[label] = []

        for contour in contours:
            label = self.PredictContour(contour)
            predictions[label].append(contour)
        
        return predictions

    def PredictContour(self, contour):
        x, y, w, h = cv.boundingRect(contour)
        contour_image = CropBoundingBox(self.__image__, x, y, w, h)

        if contour_image.shape[0] < winSize[0] or contour_image.shape[1] < winSize[1]:
            return 'others'
        # contour_image = ClusterImage(contour_image, 2)
        histogram = hog.compute(contour_image,winStride,padding,locations)

        histogram_reshape = histogram.reshape((1, histogram.shape[0] * histogram.shape[1]))

        labels = self.__model__.predict(histogram_reshape)
        return labels[0]
