## Advanced Lane Finding

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is in [CameraCal.py](https://github.com/jacquestkirk/CarND_AdvancedLaneFinding/blob/master/CameraCal.py)

Some of the calibration images cut off rows or columns, so I created 3 sets of object points. One with all 9x6 points, one with one less column, and one with one less row. 

Using glob, I interated through all the images in the calibration file. I used cv2.findChessboardCorners to find the image points. I start by trying the full set of corners, if that fails, I try with the smaller variants. If all that fails, I just give up and move to the next image. Meanwhile I collect a list of object points and image points. 

I then use cv2.calibrateCamera to get back the camera camera matrix (mtx) and distortion coefficents (dist)

Uncalibrated and calibrated test images are shown below. 

right: uncalibrated, left: calibrated
![alt text](\camera_cal\calibration3.jpg)
![alt text](\camera_cal\processed\calibration3_undistorted.jpg)

in script.py, the camera matrix and distortion coeffiecints are stored in a pickle file and reloaded, so that I don't have to re-run camera calibration every time. 

### Pipeline (single images)
The pipeline can be found in the processImage() function in [ImageProcessor.py](https://github.com/jacquestkirk/CarND_AdvancedLaneFinding/blob/master/ImageProcessor.py)


#### 1. Provide an example of a distortion-corrected image.

Undistortion occurs in the undistort function in ImageProcessor.py using cv2.undistort and the camera matrix and distortion coefficents. 

Below is an example of the original image (left) and the undistorted image (right). 

![alt text](\pipeline_images\originalImage.jpeg)
![alt text](\pipeline_images\undistorted.jpeg)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I actually did this step after doing the perspective transform, so skip to section 3 if you want to read this in order. 

See the generateMask function in ImageProcessor.py for the code that creates the binary image. 

In ImageProcessor.py I have a imageSpaceData class that I use to pull out red, green, blue, hue, saturation, lightness, and grayscale versions of the image. In addition, there is the helper Sobel_X to calculate the magnitude of the sobel in the x direction for an a given input. 

I used these functions to play around with different representation. I eventually settled on the x direction sobel of the greyscale and saturation images. The greyscale seemed to do well in shadows, while saturation did well when the white line crossed over the light concrete surface. Taking the sobel seemed to be less noisy than using magnitude. 

Below is an example of the results of thresholding. I show the greyscale sobel in green and the saturation sobel in blue. In actual processing I combinded the mask with an or function. 

![alt text](\pipeline_images\mask.jpeg)


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This step was actually done before the binary image generation. 

The transformation matrix is calculated at the bottom of [Camera.py](https://github.com/jacquestkirk/CarND_AdvancedLaneFinding/blob/master/Camera.py). The corners of the transform are given in the region_of_interest variable in Camera.py. cv2.getPerspectiveTransform is used to get the tranformation matrix and its inverse. 

The destination points are showin in the image below. 
![alt text](\pipeline_images\RegionOfInterest.jpeg)

Perspective transform was done using the getTopDown function in ImageProcessor.py. Basically it just calls cv2.warpPerspective witht the transformation matrix. The resulting image is shown below. 

![alt text](\pipeline_images\topDown.jpeg)


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are three functions that I use to find the best fit polynomial:
- **findFit** in ImageProcessor.py: This finds best fit coefficients for the current frame
- **addFilteredLine** in script.py: Keeps a runing average of past fit coefficients. Just keeps the past frame's coefficients if metrics are not met. 
- **drawFitLine** in ImageProcessor.py: Draws the filtered fit line. 

**In findFit:**

To find the best fit for a frame I first used a histogram on the bottom part of the image to look for starting points for each lane line. Then, I split the image into 9 windows stacked vertically. The midpoint of the bottom window was centered on the histogram maxes and the center of each subsequent one was located at the lower window's mean. All the points within the window were flagged. 

Then I used polyfit to get a second order fit to the flagged points. 

**In addFilteredLine:**
To determine if I had a good fit or a bad fit on the frame I set a threshold on the difference between the left and right line fit coefficients. I only placed this constraint on the quadratic and linear terms since the offset term is expected to be different between the left and right lines. 

If the frame's coefficients make it through this filter, I incorporated it into a sliding average witht the previous 10 points. 

**In drawFitLine:**
Using the filtered coefficients, I used the fit function to calculate x values along the y direction of the image. I used these points in cv2.fillPoly, to draw a shape on the road. 

An example of the resulting image is shown below. 

![alt text](\pipeline_images\fitTopDown.jpeg)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is calculated in the calculateRadiusOfCurvature function in ImageProcessor.py. It applies a fit on the non-filtered flagged points from question 4. But scales the x and y directions by xm_per_pix and ym_per_pix respectively. This changes the fit from units of pixels to units of meters. 

Then I used the equation for radius of curvature found in the lecture notes. e.g.

left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
			
Distance from the center is calculated in calculateDistanceFromCenter in ImageProcessor.py. If finds the middle of the lane by taking the average of the bottom point of the fits for the left line and the right line. This average is subtracted from the center of the image and scaled by xm_per_pix to get distance from the center in meters. This calculation assumes that the camera is mounted in the center of the car. 


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This is done in the revertPerspective function in ImageProcessor.py. Basically it calls warpPerspective on the inverse transformation matrix determined in question 3.

Furthermore, I added text noting the radius of curvature and distance from center.  

Below is an example of the results. 

![alt text](\pipeline_images\result.jpeg)

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

To generate the video run [script.py](https://github.com/jacquestkirk/CarND_AdvancedLaneFinding/blob/master/script.py)

The resulting video is stored as [project_video_processed.mp4](https://github.com/jacquestkirk/CarND_AdvancedLaneFinding/blob/master/project_video_processed.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Some of the challenging things in this project were: 
- Dealing with the concrete sections of the road. The light markers on the light road were hard to pick out. Using saturation in the mask seems to help with this.  
- Dealing with shadows. Shadows cause a change in color that the pipeline has to deal with. Using greyscale in the mask seems to help with this. 
- Bouncing. When the car went over bumps, the car, and hence the camera, bounced up and down a few times. The running average helps to keep the lines on track even if the car is angled in a different direction. 

Places where it can fail: 
- If there is another car close to the front of the camera, That car could pass the binary mask and throw off the lane line calculations. Only searching around the current lane lines might help with that. 
- Sharp curves: The center of the next stacked search window is determined from the mean of the lower window. If there is a large change, the center might not be in the center of that window. Using narrower windows might help with this. 
- Unmarked roads: If the vehicle is driving on smaller roads, they might not be marked. I don't know how to deal with this. I'd probably have to dig up an example and try stuff out. 
- Night driving: I'm not sure how the thresholds I've chosen will change during night driving. The headlight beam might create some extra lines that pass the binary mask. In addition, saturation and greyscale might not be the appropriate metrics to create the binary mask anymore. New mask types and thresholds might be chosen based on a car's light sensor. 
