import numpy as np
import cv2
import os
import imutils
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
        self._overlay_mask = None
        self._overlay = None
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
        params.minCircularity = 0.3
        params.filterByConvexity = False
        # params.minConvexity = 0.45
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

        keypoints = detector.detect(self._overlay)

        if len(keypoints) != 64:
            print("Did not able to find all the electrodes!")
            tempx, tempy = list(), list()
            for point in keypoints:
                tempx.append([point.pt[0]])
                tempy.append([point.pt[1]])
            tempx, tempy = sorted(tempx), sorted(tempy)
            xdiff, ydiff= list(), list()
            for i in range(len(tempx)):
                if i == len(tempx) - 1:
                    break
                else:
                    xdiff.append(abs(tempx[i][0] - tempx[i+1][0]))
                    ydiff.append(abs(tempy[i][0] - tempy[i+1][0]))
        
        circled = cv2.drawKeypoints(self._image, keypoints, np.array([]), (0, 255, 0),
                                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return keypoints, circled, (xdiff, ydiff)

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

    def do_kdtree(self, combined_x_y_arrays, points):
        import scipy.spatial as ss
        mytree = ss.cKDTree(combined_x_y_arrays)
        dist, indexes = mytree.query(points)
        return indexes

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

    def detect_ventricle(self):
        if self._overlay is None:
            raise Exception("No overlay mask found!")

        # masked_image = self._overlay
        # filter = cv2.Laplacian(masked_image, cv2.CV_64F)
        # ventricle = cv2.Canny(masked_image, 70, 120)
        # ventricle = cv2.Canny(self.gray, 70, 120)
        # masked_image = ventricle * m
        # blend = cv2.addWeighted(self._image, 0.5, ventricle, 0.5, 0)
        return None


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



