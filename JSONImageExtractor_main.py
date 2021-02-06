from BottleCapDetector.JSONDataExtractor.ImagePolygonExtractor import ImagePolygonExtractor
import BottleCapDetector.JSONDataExtractor.JSONDataReader as JSONDataReader
import glob, os
from datetime import datetime

tik = datetime.now()

dataset_folder = r'/home/cvvp/Projects/Computer Vision/DataSet/CV_VideoPackage'
output_folder = r'/home/cvvp/Projects/git/BottleCapDetector/Output'

file_count = len(glob.glob(os.path.join(dataset_folder, '*.json')))

for i in range(1, file_count+1):
    image_path = os.path.join(dataset_folder, 'CV20_image_{id}.png'.format(id = i))
    json_path = os.path.join(r'''/home/cvvp/Projects/Computer Vision/DataSet/CV_VideoPackage/DataSet_distractors''', 'CV20_image_{id}.json'.format(id = i))

    if not os.path.isfile(json_path):
        continue

    # Get Image Data
    try:
        DataReader = JSONDataReader.JSONDataReader(json_path)
        ImageData = DataReader.GetImageData()
    except Exception as e:
        print('An error occured in getting JSON data for ID: {id}. The exception is: {exc}'.format(id=i, exc=e))
        continue
    
    # Extract Images
    try:
        imageExtractor = ImagePolygonExtractor(image_path, ImageData, output_folder)
        imageExtractor.ExtractData()
    except Exception as e:
        print('An error occured in extracting features for ID: {id}. The exception is: {exc}'.format(id=i, exc=e))
        continue

tok = datetime.now()

print('The total time taken: ' + str(tok-tik))