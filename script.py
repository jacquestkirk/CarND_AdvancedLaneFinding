from moviepy.editor import VideoFileClip
import pickle
import cv2
import time

import CameraCal
from Camera import *
import matplotlib.pyplot as plt

showWindows = False


class imageSpaceData:
    def __init__(self, image):

        self.red = image[:, :, 0]
        self.green = image[:, :, 1]
        self.blue = image[:, :, 2]

        hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        self.hue = hls_image[:, :, 0]
        self.lightness = hls_image[:, :, 1]
        self.saturation = hls_image[:, :, 2]

        self.grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class line:
    def __init__(self, numFrames):
        self.line_history = np.zeros(shape = (numFrames, 3))
        self.radius_of_curvature = np.zeros(numFrames)
        self.center_distance = np.zeros(numFrames)
        self.index = 0

    def addToLineHistory(self, toAdd):
        self.line_history[self.index,:] = toAdd

    def addToRadiusOfCurvature(self, toAdd):
        self.radius_of_curvature[self.index] = toAdd

    def addToCenter_distance(self, toAdd):
        self.center_distance[self.index] = toAdd

    def incrementIndex(self):
        self.index += 1

def drawPolygon(image, region_of_interest, color=[255, 0, 0], thickness=5):
    x = [region_of_interest[0][0],
         region_of_interest[1][0],
         region_of_interest[2][0],
         region_of_interest[3][0]]
    y = [region_of_interest[0][1],
         region_of_interest[1][1],
         region_of_interest[2][1],
         region_of_interest[3][1]]

    # draw region of interest
    cv2.line(image, (x[0], y[0]), (x[1], y[1]), color, thickness)
    cv2.line(image, (x[1], y[1]), (x[2], y[2]), color, thickness)
    cv2.line(image, (x[2], y[2]), (x[3], y[3]), color, thickness)
    cv2.line(image, (x[3], y[3]), (x[0], y[0]), color, thickness)


def Get_HLS(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    hls_class = hlsData(hls)

    return hls_class

def Sobel_X(image):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    return scaled_sobel

def Sobel_SelectX(image, thresh = (0,255)):


    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary

def Value_Select(image, thresh=(0, 255)):
    binary_output = np.zeros_like(image)
    binary_output[(image > thresh[0]) & (image <= thresh[1])] = 1

    return binary_output



def process_image(image):

    #undistort
    undist = cv2.undistort(image, camera.mtx, camera.dst, None, camera.mtx)

    #drawPolygon(undist, camera.region_of_interest)

    #convert to top down view
    top_down = cv2.warpPerspective(undist, camera.M, camera.img_size)


    top_down_imageSpace = imageSpaceData(top_down)

    sobel_X = Sobel_X(top_down_imageSpace.saturation)
    mask_sobel_X_sat = Value_Select(sobel_X, thresh=[20, 75])
    mask_saturation = Value_Select(top_down_imageSpace.saturation, thresh=[170, 255])
    mask_sobel_X_grey = Value_Select(Sobel_X(top_down_imageSpace.grey), thresh=[25, 255])

    mask = mask_sobel_X_grey | mask_sobel_X_sat

    #thing_to_plot = mask * 255
    color_binary = np.dstack((np.zeros_like(mask_sobel_X_sat), mask_sobel_X_grey, mask_sobel_X_sat)) * 255
    #color_binary = np.dstack((thing_to_plot,thing_to_plot,thing_to_plot))

    binary_warped = mask

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one



    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if(showWindows):
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if(leftx.shape[0]<10):
        return out_img
    left_fit = np.polyfit(lefty, leftx, 2)


    if (rightx.shape[0] < 10):
        return out_img
    right_fit = np.polyfit(righty, rightx, 2)


    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))

    #calculate radius of curvature
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])



    #calculate distance from center
    center_location = np.average([left_fitx[y_eval], right_fitx[y_eval]]) - out_img.shape[0]/2
    center_location = center_location * xm_per_pix #counvert to real distance


    #unTransform fit arrays
    perspectiveLines = cv2.warpPerspective(out_img, camera.M_inverse, camera.img_size)
    result = cv2.addWeighted(undist, 1, perspectiveLines, 0.3, 0)

    leftLine.addToLineHistory(left_fit.T)
    rightLine.addToLineHistory(right_fit.T)
    leftLine.addToRadiusOfCurvature(left_curverad)
    rightLine.addToRadiusOfCurvature(right_curverad)
    leftLine.addToCenter_distance(center_location)
    leftLine.incrementIndex()
    rightLine.incrementIndex()

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



