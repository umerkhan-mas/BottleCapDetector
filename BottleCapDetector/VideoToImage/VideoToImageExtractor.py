import cv2 as cv
import numpy as np
from BottleCapDetector.Helpers.Helper import ConvertImage2GrayScale

class Video2ImageExtractor:
    def __init__(self, video, startDiscard=0.2, endDiscard=0.25, framesPerSecondCheck=6, autoDetectStartFrame=False):
        if type(video) == type(cv.VideoCapture()):
            self.__video__ = video
        elif (type(video) == type('')):
            self.__video__ = cv.VideoCapture(video)
        else:
            raise Exception('Cannot create Video2ImageExtractor. Incorrect path or video type')
                
        self.__startDiscard__ = startDiscard
        self.__endDiscard__ = endDiscard
        self.__framesPerSecondCheck__ = framesPerSecondCheck
        self.__autoDetectStartFrame__ = autoDetectStartFrame
    
    def GetVideoFrameCountAndFPS(self):
        if not self.__video__.isOpened(): 
            raise Exception("File could not be opened.")

        return int(self.__video__.get(cv.CAP_PROP_FRAME_COUNT)), int(self.__video__.get(cv.CAP_PROP_FPS))

    def GetFrameDifference(self, frame1, frame2, convertToGrayScale=True):
        if convertToGrayScale:
            frame1 = ConvertImage2GrayScale(frame1)
            frame2 = ConvertImage2GrayScale(frame2)
        frame_diff = cv.absdiff(frame1,frame2)
        return frame_diff

    def GetAbsoluteFrameDifference(self, frame1, frame2, convertToGrayScale=True):
        frame_diff = self.GetFrameDifference(frame1,frame2, convertToGrayScale)
        return np.sum(frame_diff)

    def SkipFrames(self, video, numberOfFrames=0):
        for i in range(numberOfFrames-1):
            if not video.isOpened():
                break
            video.read()

    def GetImage(self):
        image, frame_number = self.GetImageAndFrameNumber()
        return image

    def GetImageAndFrameNumber(self):
        frames, fps = self.GetVideoFrameCountAndFPS()

        # Calculate the number of frmaes to skip. We will only check divide each second into FramesPerSecondCheck and only perform differences amongst them.
        skipFrameCount = int(np.floor(fps/self.__framesPerSecondCheck__))

        frame_start = int(frames*self.__startDiscard__)
        frame_end = frames - int(frames*self.__endDiscard__ )        
        
        if self.__autoDetectStartFrame__:
            init_frame_difference = 1
            ret, previousframe = self.__video__.read()
            frame_start = 0
            while (init_frame_difference > 0.901) and (init_frame_difference < 1.009):
                self.SkipFrames(self.__video__, skipFrameCount)
                ret, currentframe = self.__video__.read()
                frame_start += skipFrameCount + 1

                if not self.__video__.isOpened():
                    raise Exception('Could not auto detect intial frame.')
                # init_frame_difference = self.GetAbsoluteFrameDifference(currentframe, previousframe)
                init_frame_difference = np.sum(currentframe) * 1.0 / np.sum(previousframe)
                previousframe = currentframe
        else:
            self.__video__.set(1, frame_start)
        current_frame_number = frame_start

        if frame_start < 0 or frame_end < 0 or frame_start >= frame_end:
            raise Exception("Invalid frame start or end. Frame start shoud not be less than 0 or greater then frame end. Frame end should also not be less than zero. Current frame_start:" + str(frame_start) + ", frame_end:" + str(frame_end))

        ret, previous_frame = self.__video__.read()

        minimum_difference = -1
        minimum_difference_image = previous_frame
        minimum_difference_frame_number = current_frame_number
        
        loop_count = (frame_end-frame_start-1)
        if skipFrameCount > 0:
            loop_count = int((frame_end-frame_start-1)/skipFrameCount)


        

        for i in range(loop_count):
            self.SkipFrames(self.__video__, skipFrameCount)
            ret, current_frame = self.__video__.read()
            current_frame_number += skipFrameCount + 1
            if not self.__video__.isOpened():
                break
            frame_difference = self.GetAbsoluteFrameDifference(current_frame, previous_frame)

            if(frame_difference < minimum_difference or minimum_difference == -1):
                minimum_difference = frame_difference
                minimum_difference_image = current_frame
                minimum_difference_frame_number = current_frame_number

            previous_frame = current_frame

        return minimum_difference_image, minimum_difference_frame_number