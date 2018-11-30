from __future__ import division

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from functools import partial
import os
import cv2

from PIL import Image
import StringIO

import numpy as np
import scipy

from skimage.morphology import skeletonize

from scipy.spatial import cKDTree
from scipy import optimize

from sparc.videotracking.optimization import Minimize
import matplotlib.pyplot as plt


class Processing:
    def __init__(self):
        self._image = None
        self._gray = None
        self._blur = None
        self._rgb = None
        self._blur_hsv = None
        self._roi_mask = None
        self._electrode_mask = None
        self._final_mask = None
        self._kernel = None
        self._bgr = None
        self._roi = None
        self._overlay_mask = None
        self._overlay = None
        self._threshold = None
        self._detected_electrodes = None
        self._reference_points = np.array([], dtype=np.float32)
        self._is_buffer = True
        self._grid = None
        self._full_detected_electrodes = None

    def get_image_size(self):
        if self._image is None:
            raise Exception("No image found! Please use read_image() method to load your image first.")
        return self._image.shape[:2]

    def get_filtered_image(self):
        return self._blur_hsv, self._gray

    def read_image(self, file_name):
        if isinstance(file_name, (bytes, bytearray)):
            pil_image = Image.open(StringIO.StringIO(file_name))
            self._image = np.array(pil_image)
        elif type(file_name) == str:
            self._image = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self._is_buffer = False
        elif type(file_name) == unicode:
            self._image = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self._is_buffer = False
        else:
            raise TypeError("Image format not supported. Only file path string or memory byte buffer are accepted.")

    def gray_and_blur(self, threshold=None):
        if self._image is None:
            raise Exception("No image selected! Please read the image first.")
        self._threshold = 5 if threshold is None else threshold

        self._gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        if self._blur is not None:
            self._blur = None
        self._blur = cv2.GaussianBlur(self._gray, (self._threshold, self._threshold), 0)
        return self._gray, self._blur

    def rgb_and_blur_and_hsv(self, threshold=None):
        if self._image is None:
            raise Exception("No image selected! Please read the image first.")
        self._threshold = 5 if threshold is None else threshold

        self._rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        if self._blur is not None:
            self._blur = None

        self._blur = cv2.GaussianBlur(self._rgb, (self._threshold, self._threshold), 0)
        self._blur_hsv = cv2.cvtColor(self._blur, cv2.COLOR_RGB2HSV)
        self._gray, self._blur = self.gray_and_blur(threshold=self._threshold)
        return self._gray

    def mask_and_image(self, roi):
        self._roi = roi
        self._roi_mask = np.zeros(self._blur.shape[:2], dtype=np.uint8)
        cv2.rectangle(self._roi_mask,
                      (self._roi[0], self._roi[1]), (self._roi[2], self._roi[3]),
                      255, thickness=-1)
        return self._roi_mask

    @staticmethod
    def electrode_boundary():
        return np.array([0, 0, 0]), np.array([15, 15, 15])

    @staticmethod
    def some_parameters():
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 20
        params.maxThreshold = 200
        params.filterByArea = True
        params.maxArea = 100
        params.minArea = 20
        params.filterByCircularity = True
        params.minCircularity = 0.45
        params.filterByConvexity = True
        params.minConvexity = 0.45
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.001
        # params.maxInertiaRatio = 1
        # params.minDistBetweenBlobs = 17
        return params

    def determine_electrode_mask(self):
        min_boundary, max_boundary = self.electrode_boundary()
        self._electrode_mask = cv2.inRange(self._blur_hsv, min_boundary, max_boundary)
        return self._electrode_mask

    def final_mask(self):
        self._final_mask = self._electrode_mask + self._roi_mask
        return self._final_mask

    @staticmethod
    def find_electrodes(input_mask):
        input_mask = input_mask.copy()
        _, contours, hierarchy = cv2.findContours(input_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        mask = np.zeros(input_mask.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, (0, 255, 0), 3)
        return biggest_contour, mask

    def overlay_mask(self, mask):
        rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        # img = cv2.addWeighted(rgb_mask, 0.5, self._rgb, 0.5, 0)
        img = cv2.addWeighted(mask, 0.5, self._gray, 0.5, 0)
        return img

    def detect_electrodes(self, kernel=None):
        self._kernel = 9 if kernel is None else kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._kernel, self._kernel))
        mask_closed = cv2.morphologyEx(self._final_mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        _, mask = self.find_electrodes(mask_clean)
        self._overlay = self.overlay_mask(mask_clean)
        params = self.some_parameters()
        ver = cv2.__version__.split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)
        masked_data = cv2.bitwise_and(self._overlay, self._overlay, mask=mask_clean)
        key_points = detector.detect(masked_data)
        self._detected_electrodes = np.asarray([key_point.pt for key_point in key_points])

        if self._is_buffer:
            self._reference_points = np.array([
                [489.205, 301.08],
                [698.477, 205.956],
                [787.137, 438.059],
                [571.962, 524.62]], dtype=np.float32)
            temp = [1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 22, 24, 25, 26, 38, 55, 62]
            self._electrode_mesh, _ = self._generate_grid()
            self._full_detected_electrodes = self._optimize(visualise=False)
        else:
            self._reference_points = np.array([
                [419.28543, 293.20193],
                [649.6535, 188.68832],
                [732.70465, 448.59],
                [509.44885, 545.94147]], dtype=np.float32)

                # [419.764, 291.567],
                # [649.012, 188.834],
                # [733.672, 449.473],
                # [506.237, 426.694]], dtype=np.float32)
            temp = [2, 3, 5, 18, 39]
            i, j = self._create_grid()
            self._electrode_mesh = np.array((i.ravel(), j.ravel())).T
            self._electrode_mesh, _ = self._generate_grid()
            self._detected_electrodes = np.delete(self._detected_electrodes, (16), axis=0)

        self._full_detected_electrodes = self._optimize(visualise=False)

        final_grid = np.zeros((64, 2))
        final_grid[:self._detected_electrodes.shape[0]] = self._detected_electrodes
        final_grid[self._detected_electrodes.shape[0]:self._detected_electrodes.shape[0]+len(self._reference_points)] =\
            self._reference_points
        ct = 0
        for pt in range(len(final_grid)):
            if pt in temp:
                final_grid[self._detected_electrodes.shape[0]+len(self._reference_points)+ct] = self._full_detected_electrodes[pt]
                ct+=1
        return final_grid, 0.0

    def _create_grid(self):
        pt1 = [self._roi[0], self._roi[1]]
        pt2 = [self._roi[2], self._roi[3]]
        x, y = np.linspace(pt1[0], pt2[0], 8), np.linspace(pt1[1], pt2[1], 8)
        xx, yy = np.meshgrid(x, y)
        return xx, yy

    @staticmethod
    def _closest_points(a, B):
        deltas = B - a
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)

    def _optimize(self, visualise=False, callback=None):

        def visualize(iteration, error, X, Y, ax):
            plt.cla()
            ax.scatter(X[:, 0], X[:, 1], color='red', label='DETECTED')
            ax.scatter(Y[:, 0], Y[:, 1], color='blue', label='GRID')
            plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error),
                     horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                     fontsize='x-large')
            ax.legend(loc='upper left', fontsize='x-large')
            plt.draw()
            plt.pause(0.001)

        if visualise:
            fig = plt.figure()
            fig.add_axes([0, 0, 1, 1])
            callback = partial(visualize, ax=fig.axes[0])
            reg = Minimize(self._detected_electrodes, self._electrode_mesh, max_iter=500, tolerance=0.1e-9)
            reg.register(callback)
            plt.show()
        else:
            callback = None
            reg = Minimize(self._detected_electrodes, self._electrode_mesh, max_iter=500, tolerance=0.1e-9)
            reg.register(callback)

        full_electrodes = reg.TY
        return full_electrodes

    def optimization(self, X, Y):
        """
        This optimization was tested and performed slower than the one currently used in draw_electrodes().

        :param X:
        :param Y:
        :return:
        """
        D = X
        T = Y
        t0 = scipy.array([0.0, 0.0, 0.0, 1.0])
        TTree = cKDTree(T)
        D = scipy.array(D)

        def obj(t):
            transformedData = self.affine_about_CoI(X, t)
            dataTree = cKDTree(transformedData)
            dD, di = dataTree.query(Y)
            dT, ti = TTree.query(transformedData)
            return (dD * dD).mean()

        def index(t):
            D1 = cKDTree(D)
            T1 = cKDTree(T)
            _, indexes1 = D1.query(X)
            _, indexes2 = T1.query(Y)
            return indexes1, indexes2

        initialRMSE = scipy.sqrt(obj(t0).mean())
        minima = []
        reses = []
        for rot in np.linspace(0, 360, 1000):
            t0 = scipy.array([0.0, 0.0, np.deg2rad(rot), 1.0])
            res = scipy.optimize.minimize(obj, t0, method='Nelder-Mead',
                                          options={'disp': False, 'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1e7})
            finalRMSE = scipy.sqrt(obj(res.x).mean())
            print(finalRMSE)
            minima.append(finalRMSE)
            reses.append(res.x)

        tOpt = reses[minima.index(min(minima))]
        optimized_points = self.affine_about_CoI(D, tOpt)
        return optimized_points

    def affine_about_CoI(self, x, t):
        image_size = self.get_image_size()
        xO = x - np.array(image_size) / 2
        xOT = Processing.affine(xO, t)
        return xOT + np.array(image_size) / 2

    @staticmethod
    def affine(x, t):
        X = scipy.vstack((x.T, scipy.ones(x.shape[0])))
        T = scipy.array([
                            [1.0, 0.0, t[0]],
                            [0.0, 1.0, t[1]],
                            [1.0, 1.0, 1.0]
                        ])
        Rx = scipy.array([
                            [scipy.cos(t[2]), -scipy.sin(t[2])],
                            [scipy.sin(t[2]), scipy.cos(t[2])]
                        ])
        T[:2, :2] = Rx
        temp = scipy.dot(T, X)[:2, :].T
        return scipy.multiply(temp, t[3])

    def _generate_grid(self):
        # initial_points = np.array([
        #                     [419.28543, 293.20193],
        #                     [649.6535, 188.68832],
        #                     [732.70465, 448.59],
        #                     [509.44885, 545.94147]
        #                     ], dtype=np.float32).tolist()
        initial_points = self._reference_points.tolist()
        p1 = initial_points[0]
        p2 = initial_points[1]
        p3 = initial_points[2]
        p4 = initial_points[3]

        number_on_side = 8
        ns = number_on_side
        ns1 = number_on_side - 1
        grid_coord = []
        for i in range(number_on_side):
            for j in range(number_on_side):
                w1 = i * j / (ns1 ** 2)  # top left point
                w2 = (j / ns1) * (ns1 - i) / ns1  # top right point
                w4 = (i / ns1) * (ns1 - j) / ns1  # The 'bottom left' point, p4
                w3 = ((ns1 - i) * (ns1 - j)) / (ns1 ** 2)  # The diagonal point, p3
                x = p4[0] * w1 + p3[0] * w2 + p2[0] * w3 + p1[0] * w4
                y = p4[1] * w1 + p3[1] * w2 + p2[1] * w3 + p1[1] * w4

                grid_coord.append([x, y])
        return np.asarray(grid_coord), np.asarray(initial_points)

    @staticmethod
    def sliding_window(arr, window_size):
        from numpy.lib.stride_tricks import as_strided
        """ Construct a sliding window view of the array"""
        arr = np.asarray(arr)
        window_size = int(window_size)
        if arr.ndim != 2:
            raise ValueError("need 2-D input")
        if not (window_size > 0):
            raise ValueError("need a positive window size")
        shape = (arr.shape[0] - window_size + 1, arr.shape[1] - window_size + 1, window_size, window_size)
        if shape[0] <= 0:
            shape = (1, shape[1], arr.shape[0], shape[3])
        if shape[1] <= 0:
            shape = (shape[0], 1, shape[2], arr.shape[1])
        strides = (arr.shape[1] * arr.itemsize, arr.itemsize, arr.shape[1] * arr.itemsize, arr.itemsize)
        return as_strided(arr, shape=shape, strides=strides)

    @staticmethod
    def get_neighbours(arr, i, j, d):
        """Return d-th neighbors of cell (i, j)"""
        w = Processing.sliding_window(arr, 2 * d + 1)

        ix = np.clip(i - d, 0, w.shape[0] - 1)
        jx = np.clip(j - d, 0, w.shape[1] - 1)

        i0 = max(0, i - d - ix)
        j0 = max(0, j - d - jx)
        i1 = w.shape[2] - max(0, d - i + ix)
        j1 = w.shape[3] - max(0, d - j + jx)
        return w[ix, jx][i0:i1, j0:j1].ravel()

    @staticmethod
    def circle_contour(image, contour):
        image_with_ellipse = image.copy()
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image_with_ellipse, ellipse, (100, 50), 2, cv2.LINE_AA)
        return image_with_ellipse

    def get_two_images(self, path, frame1, frame2):
        """
        Temporary method to read two frames.

        :param path:
        :param frame1: string name of the first image
        :param frame2: string name of the second image
        :return: gray, thresholded versions of the two frames
        """
        self.read_image(os.path.join(path, frame1))
        gray_1, _ = self.gray_and_blur(threshold=3)
        self.read_image(os.path.join(path, frame2))
        gray_2, _ = self.gray_and_blur(threshold=3)
        return gray_1, gray_2

    @staticmethod
    def get_flow(im1, im2):
        # flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 40, 11, 7, 1.5, 0)
        flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 100, 3, 71, 1.5, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        ret2, th2 = cv2.threshold(np.uint8(mag), 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th2

    def find_heart_region(self, im, kernel=None):
        if self._kernel is not None:
            self._kernel = None
        self._kernel = 5 if kernel is None else kernel
        kernel_close = np.ones((self._kernel, self._kernel), np.uint8)
        im_closed = cv2.morphologyEx(np.uint8(im), cv2.MORPH_CLOSE, kernel_close)
        kernel_open = np.ones((self._kernel*3, self._kernel*3), np.uint8)
        mask_clean = cv2.morphologyEx(im_closed, cv2.MORPH_OPEN, kernel_open)
        kernel_erode = np.ones((self._kernel*3, self._kernel*3), np.uint8)
        im_erosion = cv2.erode(mask_clean, kernel_erode, iterations=1)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im_erosion, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        min_size = 5000
        initial_heart_region = np.zeros(im.shape)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                initial_heart_region[output == i + 1] = 255
        kernel_dilate = np.ones((self._kernel*3, self._kernel*3), np.uint8)
        initial_heart_region_dilate = cv2.dilate(np.uint8(initial_heart_region), kernel_dilate, iterations=1)
        final_kernel = np.ones((self._kernel*13, self._kernel*13), np.uint8)
        closed_heart_region = cv2.morphologyEx(np.uint8(initial_heart_region_dilate), cv2.MORPH_CLOSE, final_kernel)
        blur_heart = cv2.bilateralFilter(closed_heart_region, 19, 75, 75)
        return blur_heart

    def segment_heart(self, im, mask, kernel=None, max_filter=6000, min_filter=50):
        if self._kernel is not None:
            self._kernel = None
        self._kernel = 3 if kernel is None else kernel

        segmentation = cv2.Canny(im, 70, 100)
        mask_image = cv2.bitwise_and(segmentation, segmentation, mask=mask)

        kernel = np.ones((self._kernel+10, self._kernel+10), np.uint8)
        im_closed = cv2.morphologyEx(np.uint8(mask_image), cv2.MORPH_CLOSE, kernel)
        im_closed = cv2.dilate(np.uint8(im_closed), kernel, iterations=1)
        mask_clean = cv2.morphologyEx(im_closed, cv2.MORPH_OPEN, kernel)

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        max_size = max_filter
        heart_region_inital = np.zeros(mask_clean.shape)
        for i in range(0, nb_components):
            if sizes[i] <= max_size:
                heart_region_inital[output == i + 1] = 255

        kernel = np.ones((self._kernel*3, self._kernel*3), np.uint8)
        im_erosion = cv2.erode(heart_region_inital, kernel, iterations=1)
        im_erosion = np.asarray(im_erosion, np.uint8)

        im_erosion = im_erosion.astype(np.uint8)
        im_erosion[im_erosion == 255] = int(1)
        skeleton = skeletonize(im_erosion)

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(skeleton.astype(np.uint8),
                                                                                   connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        max_size = min_filter
        heart_region = np.zeros(im_erosion.shape)
        for i in range(0, nb_components):
            if sizes[i] >= max_size:
                heart_region[output == i + 1] = 255
        return heart_region

    @staticmethod
    def draw_flow(img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    @staticmethod
    def draw_hsv(flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def feature_detect(self, h=2000, report_values=False):
        if report_values:
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(self._blur)
            print(minVal, maxVal, minLoc, maxLoc)
        surf = cv2.xfeatures2d.SURF_create(h)
        kp, dst = surf.detectAndCompute(self._gray, self._roi_mask)
        filtered_kp = [x for x in kp if not self._gray[int(x.pt[0] + 0.5), int(x.pt[1] + 0.5)] > 200]
        self.feature_image = cv2.drawKeypoints(self._image, filtered_kp, self._image)
        return filtered_kp, dst, self.feature_image

    def grab_cut(self):
        mask = np.zeros(self.image.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (300, 350, 600, 400)
        cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img = self.image * mask2[:, :, np.newaxis]
        return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    config = dict()
    config['path'] = r'/hpc/mosa004/Sparc/Heart/ImagesSmallSample'
    config['frame1'] = 'HeartVideo0001.jpg'
    config['frame2'] = 'HeartVideo0003.jpg'
    config['output_path'] = r'/hpc/mosa004/Sparc/Heart/output'

    PS = Processing()
    gray, next_gray = PS.get_two_images(config['path'], config['frame1'], config['frame2'])
    flow_image = PS.get_flow(gray, next_gray)
    heart_mask = PS.find_heart_region(flow_image, kernel=5)
    heart = PS.segment_heart(gray, heart_mask)
    plt.imshow(heart)
    plt.show()




