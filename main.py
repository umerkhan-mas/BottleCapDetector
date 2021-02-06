import cv2 as cv
import numpy as np
import os
import sys

from BottleCapDetector.VideoToImage.VideoToImageExtractor import Video2ImageExtractor
from BottleCapDetector.ROI.RegionOfInterestFinder import ROIFinder
from BottleCapDetector.Contours.ContourExtractor import ContourExtractor
from BottleCapDetector.HOG.Hog_Predictor import Hog_Predictor
from BottleCapDetector.Summer.ContourOutput import ContourOutput
from BottleCapDetector.Summer.ConsoleSummer import ConsoleOutput
from BottleCapDetector.Summer.CSVSummer import CSVOutput

video_file_path = r'''/home/cvvp/Projects/Videos/1.mp4'''
output_file_directory = r'''/home/cvvp/Projects/Videos/Outputs'''
print_debug_files = False

def main():
    if not os.path.isfile(video_file_path)  or not video_file_path.endswith('.mp4'):
        raise Exception('Invalid video file path.')

    if not os.path.isdir(output_file_directory):
        raise Exception('Invalid output directory.')


    video = cv.VideoCapture(video_file_path)

    if video is None or not video.isOpened(): 
        raise Exception("Video File could not be opened.")  

    # Extract image from video
    v2iExtractor = Video2ImageExtractor(video)

    image, frame_number = v2iExtractor.GetImageAndFrameNumber()

    if print_debug_files:
        cv.imwrite(os.path.join(output_file_directory, 'video2image.png'), image)

    if image is None or frame_number is None:
        raise('Cannot obtain image and frame number from video.')

    # Extract region of interest
    roiExtractor = ROIFinder(2)

    roiImage, roi_x, roi_y = roiExtractor.GetROI(image)    
    
    if print_debug_files:
        cv.imwrite(os.path.join(output_file_directory, 'ROI.png'), roiImage)

    if roiImage is None:
        raise Exception('Cannot obtain region of interest image.')

    # Perform image segmentation - get object contours
    cExtractor = ContourExtractor()
    contours = cExtractor.ExtractContours(roiImage)

    # Preict classes for contours
    predictor = Hog_Predictor(roiImage)
    predictions = predictor.PredictContours(contours)
    
    cp_image = roiImage.copy()

    # Outputter - Summer

    # Draw an output image
    ContourOutput().DrawContourImage(cp_image, predictions)    
    
    if print_debug_files:
        cv.imwrite(os.path.join(output_file_directory, 'contours.png'), cp_image)

    # Print output to console
    ConsoleOutput().Print(predictions)

    # Save CSV file
    CSVOutput(frame_number, roi_x, roi_y).SaveCSVFile(predictions, output_file_directory)



if __name__ == "__main__":
    video_file_path = sys.argv[1]
    output_file_directory = sys.argv[2]
    main()





