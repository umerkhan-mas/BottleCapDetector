import cv2 as cv
import numpy as np

def ConvertImage2GrayScale(image):
  return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def CropBoundingBox(image, x, y, width, height):
  return image[y:y+height,x:x+width]

def MaskImage(image, points):
  # Initialize mask
  mask = np.zeros((image.shape[0], image.shape[1]))

  # Create mask that defines the polygon of points
  cv.fillConvexPoly(mask, points, 1)
  mask = mask.astype(np.bool)

  # Create output image (untranslated)
  out = np.zeros_like(image)
  out[mask] = image[mask]

  return out

def CheckContourIntersection(cnt_ref, cnt_query, edges_only = False):
  # https://stackoverflow.com/a/61882574
  intersecting_pts = []

  ## Loop through all points in the contour
  for pt in cnt_query:
      x,y = pt[0]

      ## find point that intersect the ref contour
      ## edges_only flag check if the intersection to detect is only at the edges of the contour

      if edges_only and (cv.pointPolygonTest(cnt_ref,(x,y),True) == 0):
          intersecting_pts.append(pt[0])
      elif not(edges_only) and (cv.pointPolygonTest(cnt_ref,(x,y),True) >= 0):
          intersecting_pts.append(pt[0])

  if len(intersecting_pts) > 0:
      return True
  else:
      return False

def ClusterImage(image, K):
  Z = image.reshape((-1,3))

  # convert to np.float32
  Z = np.float32(Z)

  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)

  ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)

  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  flattened = center[label.flatten()]
  reshaped = flattened.reshape((image.shape))
  return reshaped


