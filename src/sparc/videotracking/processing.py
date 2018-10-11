import numpy as np
import cv2


# from matplotlib import pyplot as plt


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
        self.threshold = None

    # TEMPORARY ROI SELECTOR METHOD
    def select_roi(self):
        if self._image is None:
            raise Exception("ROI---No image selected! Please read the image first.")

        self.roi = cv2.selectROI(self._image)
        cv2.destroyAllWindows()
        return self.roi

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

        # retval, thresholded = cv2.threshold(self._gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow('Thresholded Image', self._gray)
        # cv2.waitKey(0)
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
        params.maxArea = 200
        params.filterByCircularity = True
        params.minCircularity = 0.3
        params.filterByConvexity = True
        params.minConvexity = 0.45
        params.filterByInertia = True
        params.minInertiaRatio = 0.2
        params.maxInertiaRatio = 1
        return params

    def detect_electrode(self):
        min_boundary, max_boudary = self.electrode_boundary()
        self._electrode_mask = cv2.inRange(self._blur_hsv, min_boundary, max_boudary)
        return self._electrode_mask

    def final_mask(self):
        self._finalmask = self._electrode_mask + self._roi_mask
        return self._finalmask

    def draw_electrodes(self, kernel=None):
        self._kernel = 15 if kernel is None else kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_closed = cv2.morphologyEx(self._finalmask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        _, mask = self.find_electrodes(mask_clean)
        overlay = self.overlay_mask(mask_clean)
        params = self.some_paramerters()
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            detector = cv2.SimpleBlobDetector(params)
        else:
            detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(overlay)

        circled = cv2.drawKeypoints(self._image, keypoints, np.array([]), (0, 255, 0),
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return keypoints, circled

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
    def circle_contour(image, contour):
        image_with_ellipse = image.copy()
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image_with_ellipse, ellipse, (100, 50), 2, cv2.LINE_AA)
        return image_with_ellipse

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

    def detect_ventricle(self, mask, show=False):
        if self.blur is None:
            raise Exception("Please create grayscale image using the 'filter_and_threshold' method.")

        m = mask
        masked_image = self.gray * m
        # filter = cv2.Laplacian(masked_image, cv2.CV_64F)
        ventricle = cv2.Canny(masked_image, 70, 120)
        # ventricle = cv2.Canny(self.gray, 70, 120)
        # masked_image = ventricle * m
        blend = cv2.addWeighted(self.gray, 0.5, ventricle, 0.5, 0)

        if show:
            cv2.imshow('Ventricle Edge', blend)
            cv2.waitKey(0)
        return None
