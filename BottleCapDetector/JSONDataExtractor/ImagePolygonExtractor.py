import numpy as np
import cv2 as cv
import os
import glob

from pathlib import Path
import BottleCapDetector.JSONDataExtractor.ImageShapesData
import BottleCapDetector.JSONDataExtractor.PolygonObjectData
from BottleCapDetector.Helpers.Helper import CropBoundingBox

class ImagePolygonExtractor:

    def __init__(self, imagePath, imageShapesData, outputFileLocation):
        self.__ImagePath__ = imagePath
        self.__ImageShapeData__ = imageShapesData
        self.__OutputFileLocation__ = outputFileLocation

    def __ReadImage__(self):
        self.__Image__ = cv.imread(self.__ImagePath__)

    def __RecenterImage__(self, polygon, image):
        # Find centroid of polygon
        (meanx, meany) = polygon.Points.mean(axis=0)

        # Find centre of image
        (cenx, ceny) = (self.__Image__.shape[1]/2, self.__Image__.shape[0]/2)

        # Make integer coordinates for each of the above
        (meanx, meany, cenx, ceny) = np.floor([meanx, meany, cenx, ceny]).astype(np.int32)

        # Calculate final offset to translate source pixels to centre of image
        (offsetx, offsety) = (-meanx + cenx, -meany + ceny)

        # Define remapping coordinates
        (mx, my) = np.meshgrid(np.arange(self.__Image__.shape[1]), np.arange(self.__Image__.shape[0]))
        ox = (mx - offsetx).astype(np.float32)
        oy = (my - offsety).astype(np.float32)

        # Translate the image to centre
        recentered_image = cv.remap(image, ox, oy, cv.INTER_LINEAR)

        return recentered_image

    def __DrawRectangle__(self, image, polygon, offsetx, offsety):
        
        # Determine top left and bottom right of translated image
        topleft = polygon.Points.min(axis=0) + [offsetx, offsety]
        bottomright = polygon.Points.max(axis=0) + [offsetx, offsety]

        # Draw rectangle
        cv.rectangle(image, tuple(topleft), tuple(bottomright), color=(255,0,0))

    def __EnsureDirectory__(self, directoryPath):
        Path(directoryPath).mkdir(parents=True, exist_ok=True)

    def __GetNextFileName__(self, fileDirectory):
        files = glob.glob(os.path.join(fileDirectory, '*.*'))
        return str(len(files))

    def __SavePolygon__(self, polygon, image):
        label = polygon.Label
        directoryPath = os.path.join(self.__OutputFileLocation__, label)
        self.__EnsureDirectory__(directoryPath)
        filePath = os.path.join(directoryPath, self.__GetNextFileName__(directoryPath) + '.png')
        cv.imwrite(filePath, image)

    def ExtractData(self):
        # Read image from the defined path
        self.__ReadImage__()   

        for polygon in self.__ImageShapeData__.Shapes:
            # The below code is inspired by the following link: https://stackoverflow.com/a/30902423
            
            x, y, w, h = cv.boundingRect(polygon.Points)
            croped_image = CropBoundingBox(self.__Image__, x, y, w, h)

            
            self.__SavePolygon__(polygon, croped_image)