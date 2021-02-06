import cv2 as cv
import numpy as np

class ContourOutput:
    def DrawContourImage(self, image, dictionary):
        
        # Blue
        self.DrawAllContours(image, dictionary['BottleCap_Deformed'], (255, 0, 0))

        # Green
        self.DrawAllContours(image, dictionary['BottleCap_FaceDown'], (0, 255, 0))

        # Red
        self.DrawAllContours(image, dictionary['BottleCap_FaceUp'], (0, 0, 255))

        # White
        self.DrawAllContours(image, dictionary['others'])

    def DrawAllContours(self, image, contours, color = (255,255,255)):
        for contour in contours:
            rect = cv.minAreaRect(contour)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(image, [box], 0, color, 2)