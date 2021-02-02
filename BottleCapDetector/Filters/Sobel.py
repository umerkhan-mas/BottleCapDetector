import cv2 as cv
import numpy as np

class SobelFilter:
    def __init__(self, scale = 3, delta = 0, ddepth = cv.CV_16S, borderType=cv.BORDER_ISOLATED):
        self.__scale__ = scale
        self.__delta__ = delta
        self.__ddepth__ = ddepth
        self.__borderType__ = borderType

    def GetSobelImage(self, image, ksize=3):
        # Gradient - X
        grad_x = cv.Sobel(image, self.__ddepth__, 1, 0, ksize=ksize, scale=self.__scale__, delta=self.__delta__, borderType=self.__borderType__)
        # Gradient - Y|
        grad_y = cv.Sobel(image, self.__ddepth__, 0, 1, ksize=ksize, scale=self.__scale__, delta=self.__delta__, borderType=self.__borderType__)
            
        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)  
        
        sobel_grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        return sobel_grad