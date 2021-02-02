import cv2 as cv
import numpy as np
import os
from BottleCapDetector.Helpers.Helper import ConvertImage2GrayScale, CropBoundingBox, MaskImage, CheckContourIntersection
from BottleCapDetector.Filters.Sobel import SobelFilter

class ROIFinder:
    def __init__(self, k_color = 2):
        self.__sobelFilter__ = SobelFilter()
        self.__kColor__ = k_color
    
    def ColorQuantize(self, image, K):
        Z = image.reshape((-1,3))

        # convert to np.float32
        Z = np.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        flattened = center[label.flatten()]
        reshaped = flattened.reshape((image.shape))
        return reshaped

    def GetImageCenterBoundingBox(self, image):
        y = int(image.shape[0] / 2)
        x = int(image.shape[1] / 2)
        return x, y, 70, 70

    def GetImageCenterPoints(self, image):
        x,y,w,h = self.GetImageCenterBoundingBox(image)
        return np.array([ [[x, y]], [[x + h, y]], [[x + h, y + w]], [[x, y + w]] ])
    
    def GetROI(self,img):
        image = img.copy()
        image = self.ColorQuantize(image, self.__kColor__)
        image = ConvertImage2GrayScale(image)

        edge_image = self.__sobelFilter__.GetSobelImage(image, ksize=3)

        ret,thresh_image = cv.threshold(edge_image,50,255,0)
        contours, hierarchy = cv.findContours(thresh_image, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        image_area = image.shape[0] * image.shape[1]
        max_contour = None
        max_area = 0
        
        image_center_x, image_center_y, image_center_w, image_center_h = self.GetImageCenterBoundingBox(image)
        image_center_box = self.GetImageCenterPoints(image)

        for contour in contours:
            hull = cv.convexHull(contour)
            contour_area = cv.contourArea(hull)
            
            if contour_area >= 0.20*image_area and contour_area <= 0.7*image_area and CheckContourIntersection(contour, image_center_box):
                if contour_area > max_area:
                    max_area = contour_area
                    max_contour = contour

        if max_contour is not None:
            hull = cv.convexHull(max_contour)
            # rect = cv.minAreaRect(max_contour)
            # box = cv.boxPoints(rect)
            # box = np.int0(box)
            x, y, w, h = cv.boundingRect(max_contour)
            masked_image = MaskImage(img, hull)
            return CropBoundingBox(masked_image, x, y, w, h)
        
        return img





    