import cv2
import numpy as np


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

class ImageProcessor:

    def __init__(self, camera, leftLine, rightLine, showWindows = False):




        self.image = None
        self.camera = camera
        self.leftLine = leftLine
        self.rightLine = rightLine
        self.showWindows = showWindows

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        self.undist = None
        self.top_down = None
        self.color_binary = None
        self.fitted_top_down = None
        self.revertedPerspective = None
        self.result = None

        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.right_fit = None
        self.left_fit = None


        return

    def loadImage(self, image):
        self.image = image

    def processImage(self):
        image = self.image
        self.undist = self.undistort(image)
        self.regionOfInterest = ImageProcessor.drawPolygon(self.undist, self.camera.region_of_interest)
        self.top_down = self.getTopDown(self.undist)
        self.mask = self.generateMask(self.top_down)
        self.fitted_top_down = self.findFit(self.mask)


        [left_curverad, right_curverad] = self.calculateRadiusOfCurvature()
        center_location = self.calculateDistanceFromCenter()

        self.leftLine.addToLineHistory(self.left_fit.T)
        self.rightLine.addToLineHistory(self.right_fit.T)
        self.leftLine.addToRadiusOfCurvature(left_curverad)
        self.rightLine.addToRadiusOfCurvature(right_curverad)
        self.leftLine.addToCenter_distance(center_location)
        self.leftLine.incrementIndex()
        self.rightLine.incrementIndex()

        self.filtered_top_down = self.drawFitLine(self.fitted_top_down)

        self.revertedPerspective = self.revertPerspective(self.filtered_top_down)
        self.result = cv2.addWeighted(self.undist, 1, self.revertedPerspective, 0.3, 0)

        return self.result



    def undistort(self, image):
        undist = cv2.undistort(image, self.camera.mtx, self.camera.dst, None, self.camera.mtx)
        return undist

    def getTopDown(self, image):
        top_down = cv2.warpPerspective(image, self.camera.M, self.camera.img_size)
        return top_down

    def generateMask(self, image):
        top_down_imageSpace = imageSpaceData(image)

        sobel_X = ImageProcessor.Sobel_X(top_down_imageSpace.saturation)
        mask_sobel_X_sat = ImageProcessor.Value_Select(sobel_X, thresh=[20, 75])
        mask_sobel_X_grey = ImageProcessor.Value_Select(ImageProcessor.Sobel_X(top_down_imageSpace.grey), thresh=[25, 255])

        mask = mask_sobel_X_grey | mask_sobel_X_sat

        self.color_binary = np.dstack((np.zeros_like(mask_sobel_X_sat), mask_sobel_X_grey, mask_sobel_X_sat)) * 255

        return mask

    def findFit(self, image):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[image.shape[0] / 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(image.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
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
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if (self.showWindows):
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
        self.leftx = nonzerox[left_lane_inds]
        self.lefty = nonzeroy[left_lane_inds]
        self.rightx = nonzerox[right_lane_inds]
        self.righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        if (self.leftx.shape[0] < 10):
            return out_img
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)

        if (self.rightx.shape[0] < 10):
            return out_img
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        # Generate x and y values for plotting
        self.ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]

        #color selected points
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        return out_img

    def drawFitLine(self, image):
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(image, np.int_([pts]), (0, 255, 0))
        return image

    def revertPerspective(self, image):
        # unTransform fit arrays
        perspectiveLines = cv2.warpPerspective(image, self.camera.M_inverse, self.camera.img_size)

        return perspectiveLines

    def calculateRadiusOfCurvature(self):

        # calculate radius of curvature
        y_eval = np.max(self.ploty)


        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.lefty * self.ym_per_pix, self.leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.righty * self.ym_per_pix, self.rightx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        return[left_curverad, right_curverad]

    def calculateDistanceFromCenter(self):

        y_eval = np.max(self.ploty)

        # calculate distance from center
        center_location = np.average([self.left_fitx[y_eval], self.right_fitx[y_eval]]) - self.image.shape[0] / 2
        center_location = center_location * self.xm_per_pix  # counvert to real distance

        return center_location





    def Get_HLS(image):
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        hls_class = imageSpaceData(hls)

        return hls_class

    def Sobel_X(image):
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        return scaled_sobel

    def Sobel_SelectX(image, thresh=(0, 255)):
        # Threshold x gradient
        sxbinary = np.zeros_like(image)
        sxbinary[(image >= thresh[0]) & (image <= thresh[1])] = 1

        return sxbinary

    def Value_Select(image, thresh=(0, 255)):
        binary_output = np.zeros_like(image)
        binary_output[(image > thresh[0]) & (image <= thresh[1])] = 1

        return binary_output

    def drawPolygon(image, region_of_interest, color=[255, 0, 0], thickness=5):

        imageToDrawOn = image.copy()

        x = [region_of_interest[0][0],
             region_of_interest[1][0],
             region_of_interest[2][0],
             region_of_interest[3][0]]
        y = [region_of_interest[0][1],
             region_of_interest[1][1],
             region_of_interest[2][1],
             region_of_interest[3][1]]

        # draw region of interest
        cv2.line(imageToDrawOn, (x[0], y[0]), (x[1], y[1]), color, thickness)
        cv2.line(imageToDrawOn, (x[1], y[1]), (x[2], y[2]), color, thickness)
        cv2.line(imageToDrawOn, (x[2], y[2]), (x[3], y[3]), color, thickness)
        cv2.line(imageToDrawOn, (x[3], y[3]), (x[0], y[0]), color, thickness)
