import json
import numpy as np

from BottleCapDetector.JSONDataExtractor.ImageShapesData import ImageShapesData
from BottleCapDetector.JSONDataExtractor.PolygonObjectData import PolygonObjectData

class JSONDataReader:
    def __init__(self, filePath):
        self.__FilePath__ = filePath
        self.__ReadJSONFile__()
        self.__SetJSONData__()

    def __ReadJSONFile__(self):
        with open(self.__FilePath__) as f:
            self.__JSONData__ = json.load(f)

    def __SetJSONData__(self):
        json_shapes = self.__JSONData__['shapes']

        shapes = []
        for i in range(len(json_shapes)):
            json_shape = json_shapes[i]
            label = json_shape['label']
            points = np.array(json_shape['points'], dtype='int32')
            #points = json_shape['points']
            shapes.append(PolygonObjectData(label, points))

        imageName = self.__JSONData__['imagePath']
        #self.__ImageData__ = ImageData(imageName, shapes)
        self.__ImageData__ = ImageShapesData(imageName, np.array(shapes))

    def GetImageData(self):
        return self.__ImageData__

