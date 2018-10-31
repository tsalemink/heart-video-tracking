from __future__ import division

from numpy import *
import numpy as np
import os, sys
import cv2
from optimization import Minimize
import imutils


class Processing:
    def __init__(self):
        self._image = None
        self._gray = None
        self._blur = None
        self._rgb = None
        self._blur_hsv = None
        self._roi_mask = None
        self._electrode_mask = None
        self._finalmask = None
        self._kernel = None
        self._bgr = None
        self.roi = None
        self._overlay_mask = None
        self._overlay = None
        self.threshold = None
        self._image_size = None
        self._detected_electrodes = None
        self._grid = None
        self._full_detected_electrodes = None

    # TEMPORARY ROI SELECTOR METHOD
    def select_roi(self):
        if self._image is None:
            raise Exception("ROI---No image selected! Please read the image first.")

        self.roi = cv2.selectROI(self._image)
        cv2.destroyAllWindows()
        return self.roi

    def get_image_size(self):
        if self._image is None:
            raise Exception("No image found! Please use read_image() method to load your image first.")
        return self._image.shape[:2]

    def get_filtered_image(self):
        return self._blur_hsv, self._gray

    def read_image(self, file_name):
        self._image = cv2.imread(file_name, 1)

    def gray_and_blur(self, threshold=None):
        if self._image is None:
            raise Exception("No image selected! Please read the image first.")
        self.threshold = 5 if threshold is None else threshold

        self._gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
        if self._blur is not None:
            self._blur = None
        self._blur = cv2.GaussianBlur(self._gray, (self.threshold, self.threshold), 0)
        return self._gray, self._blur

    def rgb_and_blur_and_hsv(self, threshold=None):
        if self._image is None:
            raise Exception("No image selected! Please read the image first.")
        self.threshold = 5 if threshold is None else threshold

        self._rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        if self._blur is not None:
            self._blur = None

        self._blur = cv2.GaussianBlur(self._rgb, (self.threshold, self.threshold), 0)
        self._blur_hsv = cv2.cvtColor(self._blur, cv2.COLOR_RGB2HSV)
        self._gray, self._blur = self.gray_and_blur(threshold=9)
        return self._gray

    def mask_and_image(self, roi):
        r = roi
        self._roi_mask = np.zeros(self._blur.shape[:2], dtype=np.uint8)
        cv2.rectangle(self._roi_mask, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), 255, thickness=-1)
        return self._roi_mask

    @staticmethod
    def electrode_boundary():
        return np.array([0, 0, 0]), np.array([15, 15, 15])

    @staticmethod
    def some_paramerters():
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 90;
        params.maxThreshold = 200;
        params.filterByArea = True
        params.maxArea = 1000
        params.minArea = 20
        params.filterByCircularity = True
        params.minCircularity = 0.45
        params.filterByConvexity = False
        # params.minConvexity = 0.45
        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        params.maxInertiaRatio = 1
        params.filterByColor = True
        params.blobColor = 0
        return params

    def detect_electrode(self):
        min_boundary, max_boudary = self.electrode_boundary()
        self._electrode_mask = cv2.inRange(self._blur_hsv, min_boundary, max_boudary)
        return self._electrode_mask

    def final_mask(self):
        self._finalmask = self._electrode_mask + self._roi_mask
        return self._finalmask

    def draw_electrodes(self, kernel=None):
        from matplotlib import pyplot as plt
        from functools import partial

        self._kernel = 15 if kernel is None else kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self._kernel, self._kernel))
        mask_closed = cv2.morphologyEx(self._finalmask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        _, mask = self.find_electrodes(mask_clean)
        self._overlay = self.overlay_mask(mask_clean)

        params = self.some_paramerters()
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)
        cv2.circle(self._overlay, (mask.shape[0], mask.shape[1]), 280, 1, thickness=-1)
        masked_data = cv2.bitwise_and(self._overlay, self._overlay, mask=mask_clean)
        keypoints = detector.detect(masked_data)
        circled = cv2.drawKeypoints(self._image, keypoints, np.array([]), (0, 255, 0),
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        self._detected_electrodes = np.asarray([key_point.pt for key_point in keypoints])
        detected_electrode_array_size = len(self._detected_electrodes)

        self._full_detected_electrodes = self.optimize(visualize=True)

        self._detected_electrodes = self._detected_electrodes[np.argsort(self._detected_electrodes[:,0])]
        self._full_detected_electrodes = self._full_detected_electrodes[np.argsort(self._full_detected_electrodes[:,0])]

        self._full_detected_electrodes[:detected_electrode_array_size] = self._detected_electrodes

        # self._full_detected_electrodes = self.optimization(self._grid, self._detected_electrodes)
        return self._full_detected_electrodes, circled

    def optimize(self, visualize=False, callback=None):

        def visualization(iteration, error, X, Y, ax):
            plt.cla()
            ax.scatter(X[:, 0], X[:, 1], color='red')
            ax.scatter(Y[:, 0], Y[:, 1], color='blue')
            plt.draw()
            print("iteration %d, error %.10f" % (iteration, error))
            plt.pause(0.001)

        if visualize:
            from matplotlib import pyplot as plt
            from functools import partial
            fig = plt.figure()
            fig.add_axes([0, 0, 1, 1])
            callback = partial(visualization, ax=fig.axes[0])

        self._grid = np.asarray(self.generate_grid())
        reg = Minimize(self._detected_electrodes, self._grid, max_iter=10000, tolerance=0.000001)
        reg.tolerance = 1e-9
        reg.register(callback)
        return reg.TY

    def optimization(self, X, Y):
        import scipy
        from scipy.spatial import distance as dist
        from scipy.spatial import cKDTree
        from scipy.optimize import leastsq, fmin
        from scipy import optimize
        from scipy.linalg import lstsq

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
        self._image_size = self.get_image_size()
        xO = x - array(self._image_size) / 2
        xOT = Processing.affine(xO, t)
        return xOT + array(self._image_size) / 2

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
        img = cv2.addWeighted(rgb_mask, 0.5, self._rgb, 0.5, 0)
        return img

    @staticmethod
    def make_8by8_grid():
        x = np.linspace(-4, 4, 8)
        y = np.linspace(-4, 4, 8)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        return xx, yy

    @staticmethod
    def generate_grid():
        pointsList = np.array([
                            [419.28543, 293.20193],
                            [649.6535, 188.68832],
                            [732.70465, 448.59],
                            [509.44885, 545.94147]
                            ], dtype=np.float32).tolist()

        p1 = pointsList[0]
        p2 = pointsList[1]
        p3 = pointsList[2]
        p4 = pointsList[3]

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
        return grid_coord

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

    def get_two_images(self):
        """
        Temporarily return two hard-coded images
        :return:
        """
        path = r'/hpc/mosa004/Sparc/Heart/ImagesSmallSample'
        self.read_image(os.path.join(path, 'HeartVideo0001.jpg'))
        gray_1, blur_1 = self.gray_and_blur(threshold=9)

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


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


if __name__ == '__main__':
    ps = Processing()
    ps.generateGridPoints4()

    path = r'/hpc/mosa004/Sparc/Heart/ImagesSmallSample'
    output_path = r'/hpc/mosa004/Sparc/Heart/output'

    PS = Processing()
    im1 = os.path.join(path, 'HeartVideo0001.jpg')
    PS.read_image(im1)
    gray, blur = PS.gray_and_blur(threshold=9)

    del PS
    PS = Processing()
    im2 = os.path.join(path, 'HeartVideo0003.jpg')
    PS.read_image(im2)
    next_gray, next_blur = PS.gray_and_blur(threshold=9)

    flow = cv2.calcOpticalFlowFarneback(gray, next_gray, None, 0.5, 3, 40, 11, 7, 1.5, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    ret2, th2 = cv2.threshold(np.uint8(mag), 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_close = np.ones((5, 5), np.uint8)
    img_closed = cv2.morphologyEx(np.uint8(th2), cv2.MORPH_CLOSE, kernel_close)
    kernel_open = np.ones((13, 13), np.uint8)
    mask_clean = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel_open)
    kernel_erode = np.ones((15, 15), np.uint8)
    img_erosion = cv2.erode(mask_clean, kernel_erode, iterations=1)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_erosion, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 5000
    t2 = np.zeros(th2.shape)
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            t2[output == i + 1] = 255

    kernel_1 = np.ones((15, 15), np.uint8)
    t2_dilate = cv2.dilate(np.uint8(t2), kernel_1, iterations=1)

    kernel_close_1 = np.ones((65, 65), np.uint8)
    img_closed_1 = cv2.morphologyEx(np.uint8(t2_dilate), cv2.MORPH_CLOSE, kernel_close_1)

    blur = cv2.bilateralFilter(img_closed_1, 19, 75, 75)
    edge_OF = cv2.Canny(blur, 100, 200)
    blend = cv2.addWeighted(gray, 0.5, edge_OF, 0.5, 0)

    # cv2.imshow("mag", mag)
    # cv2.imshow("th2", th2)
    # cv2.imshow("img_closed", img_closed)
    # cv2.imshow("mask_clean", mask_clean)
    # cv2.imshow("img_erosion", img_erosion)
    # cv2.imshow("t2", t2)
    # cv2.imshow("t2_dilate", t2_dilate)
    # cv2.imshow("img_closed_1", img_closed_1)
    # cv2.imshow("blur", blur)
    # cv2.imshow("edge_OF", edge_OF)
    # cv2.imshow("blend", blend)
    # k = cv2.waitKey(20) & 0xFF

    save_path = os.path.join(output_path, "segmentation")
    cv2.imwrite(os.path.join(save_path, "gray.jpg"), next_gray)
    cv2.imwrite(os.path.join(save_path, "th2.jpg"), th2)
    cv2.imwrite(os.path.join(save_path, "img_closed.jpg"), img_closed)
    cv2.imwrite(os.path.join(save_path, "mask_clean.jpg"), mask_clean)
    cv2.imwrite(os.path.join(save_path, "img_erosion.jpg"), img_erosion)
    cv2.imwrite(os.path.join(save_path, "t2.jpg"), t2)
    cv2.imwrite(os.path.join(save_path, "t2_dilate.jpg"), t2_dilate)
    cv2.imwrite(os.path.join(save_path, "img_closed_1.jpg"), img_closed_1)
    cv2.imwrite(os.path.join(save_path, "blur.jpg"), blur)
    cv2.imwrite(os.path.join(save_path, "edge_OF.jpg"), edge_OF)
    cv2.imwrite(os.path.join(save_path, "blend.jpg"), blend)

    print("Done!")



