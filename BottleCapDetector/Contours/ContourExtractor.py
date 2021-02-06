import cv2 as cv
import numpy as np
from BottleCapDetector.Helpers.Helper import ConvertImage2GrayScale, ClusterImage
from BottleCapDetector.Filters.Sobel import SobelFilter

class ContourExtractor:

    def __init__(self):
        self.__sobelFilter__ = SobelFilter()

    def ExtractContours(self, img):
        image = img.copy()
        # image = ClusterImage(image, 2)
        image = ConvertImage2GrayScale(image)
        
        edge_image = self.__sobelFilter__.GetSobelImage(image, ksize=3)

        ret,thresh_image = cv.threshold(edge_image,100,255,0)
        contours, hierarchy = cv.findContours(thresh_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        return_contours = []

        for contour in contours:
            hull = cv.convexHull(contour)
            contour_area = cv.contourArea(hull)
            
            if contour_area >= 32*32 and contour_area<=80*80:
                return_contours.append(contour)

        return return_contours

