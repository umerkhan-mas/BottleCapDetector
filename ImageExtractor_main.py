import cv2 as cv
import glob, os
from datetime import datetime

from BottleCapDetector.VideoToImage.VideoToImageExtractor import Video2ImageExtractor

video_directory = r'''/home/cvvp/Projects/Computer Vision/DataSet/CV_VideoPackage'''
output_directory = r'''/home/cvvp/Projects/git/BottleCapDetector/BottleCapDetector/Output'''
file_count = len(glob.glob(os.path.join(video_directory, '*.mp4')))

tik = datetime.now()

for i in range(1, file_count+1):
    input_video_path = os.path.join(video_directory, 'CV20_video_{id}.mp4'.format(id = i))
    output_image_name = 'CV20_image_{id}.png'.format(id = i)

    video = cv.VideoCapture(input_video_path)
    if not video.isOpened(): 
        raise Exception("File could not be opened.")    

    v2iExtractor = Video2ImageExtractor(video)

    image, frame_number = v2iExtractor.GetImageAndFrameNumber()

    cv.imwrite(os.path.join(output_directory, output_image_name), image)

    print(str(i) + '-' + str(frame_number))

tok = datetime.now()

print('The total time taken: ' + str(tok-tik))