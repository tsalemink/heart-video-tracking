import os
import numpy as np
import cv2


class Processing:
    def __init__(self, path, filename):
        self.path = path
        self.fname = filename
        self.imfile = os.path.join(self.path, self.fname)

        self.image = None
        self.roi = None
        self.gray = None
        self.blur = None

        self.mask = None
        self.feature_image = None

    def read_image(self):
        self.image = cv2.imread(self.imfile, 1)

        return self.image

    def select_roi(self):
        if self.image is None:
            raise Exception("No image selected! Please read the image first.")
        self.roi = cv2.selectROI(self.image)
        cv2.destroyAllWindows()
        return self.roi

    def filter_and_threshold(self):
        if self.image is None:
            raise Exception("No image selected! Please read the image first.")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.GaussianBlur(self.gray, (5, 5), 0)

        return self.gray, self.blur

    def mask_and_image(self, roi):
        r = roi
        self.mask = np.zeros(self.blur.shape[:2], dtype=np.uint8)
        cv2.rectangle(self.mask, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255), thickness=-1)

        return self.mask

    def feature_detect(self, h=2000):

        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(self.blur)

        surf = cv2.xfeatures2d.SURF_create(h)
        kp, dst = surf.detectAndCompute(self.blur, self.mask)

        filtered_kp = [x for x in kp if not self.blur[int(x.pt[0]), int(x.pt[1])] > 80]

        self.feature_image = cv2.drawKeypoints(self.image, filtered_kp, self.image)

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


