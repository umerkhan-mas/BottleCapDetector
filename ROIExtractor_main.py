import cv2 as cv
import glob, os
from datetime import datetime

from BottleCapDetector.ROI.RegionOfInterestFinder import ROIFinder

input_directory = r'''/home/cvvp/Projects/Computer Vision/DataSet/CV_VideoPackage'''
output_directory = r'''/home/cvvp/Projects/git/BottleCapDetector/BottleCapDetector/Output'''
file_count = len(glob.glob(os.path.join(input_directory, '*.png')))

tik = datetime.now()

for i in range(1, file_count+1):
    file_name = 'CV20_image_{id}.png'.format(id = i)
    input_image_path = os.path.join(input_directory, file_name)

    image = cv.imread(input_image_path) 

    roiExtractor = ROIFinder(2)

    roiImage = roiExtractor.GetROI(image)

    cv.imwrite(os.path.join(output_directory, file_name), roiImage)

    print(i)


tok = datetime.now()

print('The total time taken: ' + str(tok-tik))