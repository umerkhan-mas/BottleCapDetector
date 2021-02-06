import cv2 as cv
import numpy as np
import os


class CSVOutput:
    def __init__(self, frame_number, roi_x, roi_y):
        self.__framenumber__ = frame_number
        self.__roiX__ = roi_x
        self.__roiY__ = roi_y

    def GetCSVString(self, dictionary):
        csv_file = ''
        for label, contours in dictionary.items():
            if label == 'others':
                continue

            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)

                csv_file += "{frame_number},{x_cord},{y_cord},'{label}'\n".format(
                    frame_number = self.__framenumber__,
                    x_cord = self.__roiX__ + x,
                    y_cord = self.__roiY__ + y,
                    label = label) 

        return csv_file

    def SaveCSVFile(self, dictionary, output_directory, filename='output.csv'):
        csv_file = self.GetCSVString(dictionary)
        output_file_path = os.path.join(output_directory, filename)

        with open(output_file_path, 'w+') as f:
            f.write(csv_file) 



