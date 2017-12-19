import cv2
import numpy as np


class Camera:
    def __init__(self, mtx, dst):
        self.mtx = mtx
        self.dst = dst
        self.resolution = (720, 1280, 3)
        self.max_y = 670
        self.min_y = 450
        self.width_near = 600
        self.width_far = 75

        half_height = int(self.resolution[0]/2)
        half_width = int(self.resolution[1]/2)

        corner_lower_left = [half_width - self.width_near, self.max_y]
        corner_lower_right = [half_width + self.width_near, self.max_y]
        corner_upper_left = [half_width - self.width_far, self.min_y]
        corner_upper_right = [half_width + self.width_far, self.min_y]

        self.region_of_interest = [corner_lower_left, corner_lower_right, corner_upper_right, corner_upper_left]

        self.img_size = (self.resolution[1], self.resolution[0])

        src_pts = np.array(self.region_of_interest, np.float32)
        offset = 50
        dst_pts = np.array([[offset, self.resolution[0] - offset],
                            [self.resolution[1] - offset, self.resolution[0] - offset],
                            [self.resolution[1] - offset, offset],
                            [offset, offset]], np.float32)

        self.M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inverse = cv2.getPerspectiveTransform(dst_pts, src_pts)
