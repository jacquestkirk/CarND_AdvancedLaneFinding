

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def calibrateCamera(generate_images = False):
    # prepare object points
    nx = 9
    ny = 6

    objpoints = []
    imgpoints = []

    new_objpoints = np.zeros((nx*ny,3), np.float32)
    new_objpoints[: , :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    new_objpoints_short_y = np.zeros((nx*(ny-1),3), np.float32)
    new_objpoints_short_y[: , :2] = np.mgrid[0:nx, 0:(ny-1)].T.reshape(-1,2)

    new_objpoints_short_x = np.zeros(((nx-1)*ny,3), np.float32)
    new_objpoints_short_x[: , :2] = np.mgrid[0:nx-1, 0:(ny)].T.reshape(-1,2)

    # Make a list of calibration images
    file_names = glob.glob('camera_cal//calibration*.jpg') #'camera_cal//calibration1.jpg'

    print(file_names)

    for file_name in file_names:
        img = cv2.imread(file_name)

        #plt.imshow(img)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        #print(ret)
        #print(corners)

        # If found, draw corners
        if ret == True:
            # Draw and display the corners
            objpoints.append(new_objpoints)
            imgpoints.append(corners)

            if generate_images:
                fig = plt.figure()
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        else:
            #try again with one less row in case it is cut off
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny-1), None)

            if ret == True:
                # Draw and display the corners
                objpoints.append(new_objpoints_short_y)
                imgpoints.append(corners)

                if generate_images:
                    fig = plt.figure()
                    cv2.drawChessboardCorners(img, (nx, ny-1), corners, ret)
            else:
                # try again with one less column in case it is cut off
                ret, corners = cv2.findChessboardCorners(gray, (nx-1, ny), None)
                if ret == True:
                    # Draw and display the corners
                    objpoints.append(new_objpoints_short_x)
                    imgpoints.append(corners)
                    if generate_images:
                        fig = plt.figure()
                        cv2.drawChessboardCorners(img, (nx-1, ny), corners, ret)
                else:
                    print(file_name + ' failed finding corners')
                continue

        print(file_name + ' corners found sucessfully')

        if generate_images:
            plt.imshow(img)
            file_only_name = file_name[0:-4].split('\\')[-1]
            fig.savefig('camera_cal\\processed\\' + file_only_name + '_corners.jpg')

        et, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if generate_images:
        for file_name in file_names:
            img = cv2.imread(file_name)
            undist = cv2.undistort(img, mtx, dist, None, mtx)

            plt.imshow(undist)
            file_only_name = file_name[0:-4].split('\\')[-1]
            fig.savefig('camera_cal\\processed\\' + file_only_name + '_undistorted.jpg')

            print(file_name + ' undistorted')

    return[mtx , dist]
