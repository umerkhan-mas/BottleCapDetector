import cv2 as cv

def ConvertImage2GrayScale(image):
  return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def CropBoundingBox(box):
    pass