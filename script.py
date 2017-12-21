from moviepy.editor import VideoFileClip
import pickle
import cv2
import time

import CameraCal
from Camera import *
from ImageProcessor import *
import matplotlib.pyplot as plt



class line:
    def __init__(self, numFrames):

        self.averageWindow = 10         #length of an average window
        self.max_delta_fit0 = 0.0006    #biggest difference between left and right lines for quadratic term
        self.max_delta_fit1 = 0.6       #biggest difference between left and right lines for linear term
        self.sat_radius_of_curvature = 3000 #saturate radius of curvature calculation at this point since straight roads are infinite

        self.line_history = np.zeros(shape = (numFrames, 3))
        self.radius_of_curvature = np.zeros(numFrames)
        self.center_distance = np.zeros(numFrames)

        self.line_history_filtered = np.zeros(shape=(numFrames, 3))
        self.radius_of_curvature_filtered = np.zeros(numFrames)
        self.center_distance_filtered = np.zeros(numFrames)

        self.index = 0

    def addToLineHistory(self, toAdd):
        self.line_history[self.index,:] = toAdd

    def addFilteredLine(self, delta_fit0, delta_fit1):

        if((delta_fit0 > self.max_delta_fit0)
           & (delta_fit1 > self.max_delta_fit1)
           & self.index != 0):
            #If our limits for max difference in fit variables is not met
            #take the last values
            self.line_history_filtered[self.index] = self.line_history_filtered[self.index-1]
        else:
            self.runningAverage(self.line_history_filtered, self.line_history)



    def addToRadiusOfCurvature(self, toAdd):
        self.radius_of_curvature[self.index] = toAdd
        self.runningAverage(self.radius_of_curvature_filtered, self.radius_of_curvature)

    def addToCenter_distance(self, toAdd):
        self.center_distance[self.index] = toAdd
        self.runningAverage(self.center_distance_filtered, self.center_distance)


    def incrementIndex(self):
        self.index += 1

    def runningAverage(self, previous_average_list, currentValue_list):

        averageWindow = self.averageWindow
        if (self.index < self.averageWindow):
            averageWindow = self.index

        if(self.index==0):
            new_average = currentValue_list[0]
        else:
            previous_average = previous_average_list[self.index-1]
            currentValue = currentValue_list[self.index]
            new_average = previous_average - previous_average/averageWindow + currentValue/averageWindow

        previous_average_list[self.index] = new_average





def process_image(image):

    imageProcessor.loadImage(image)
    result = imageProcessor.processImage()

    return result


cameraCal = False

if cameraCal:
    [mtx, dist] = CameraCal.calibrateCamera()
    pickle.dump([mtx, dist], open("calibration.p", "wb"))

[mtx, dist] =pickle.load( open( "calibration.p", "rb" ) )

camera = Camera(mtx, dist)

start_time = 0
end_time = 50
num_Frames = (end_time - start_time) * 25 +1

leftLine = line(num_Frames)
rightLine = line(num_Frames)
imageProcessor = ImageProcessor(camera, leftLine, rightLine)

clip1 = VideoFileClip("project_video.mp4").subclip(start_time,end_time)#.subclip(21,24)#.subclip(40,43)


newClip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
newClip.write_videofile('project_video_processed.mp4', audio=False)

time = np.linspace(0, (end_time - start_time), leftLine.center_distance.__len__())

fig1, (ax_fit0, ax_fit1, ax_fit2, ax_roc, ax_center) = plt.subplots(5, 1, sharex=True)


ax_fit0.plot(time, leftLine.line_history[:, 0], color='r')
ax_fit0.plot(time, rightLine.line_history[:, 0], color='b')
ax_fit0.plot(time, np.abs(leftLine.line_history[:, 0] - rightLine.line_history[:, 0]), color='g')
ax_fit0.set_ylabel('fit0')

ax_fit1.plot(time, leftLine.line_history[:, 1], color='r')
ax_fit1.plot(time, rightLine.line_history[:, 1], color='b')
ax_fit1.plot(time, np.abs(leftLine.line_history[:, 1] - rightLine.line_history[:, 1]), color='g')
ax_fit1.set_ylabel('fit1')

ax_fit2.plot(time, leftLine.line_history[:, 2], color='r')
ax_fit2.plot(time, rightLine.line_history[:, 2], color='b')
ax_fit2.set_ylabel('fit2')

ax_roc.plot(time, leftLine.radius_of_curvature, color='r')
ax_roc.plot(time, rightLine.radius_of_curvature, color='b')
ax_roc.set_ylabel('Radius Of Curvature')

ax_center.plot(time, leftLine.center_distance, color='b')
ax_center.set_ylabel('Center')



