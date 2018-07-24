import os, time
import numpy as np
import cv2

from Processing import Processing
from LKOpticalFlow import LKOpticalFlow


class CoordinateStore:
    """
    This class can be used to extract image coordinates
    using mouse clicks.

    Double mouse click must be used.
    """
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(im, (x, y), 4, (255, 0, 0), -1)
            self.points.append((x, y))


path = './images'
fname = 'HeartVideo0001.jpg'
PS = Processing(path, fname)

path = './HeartVideoFrames'
animation = True
count = 1

CS = CoordinateStore()

lk = LKOpticalFlow(win=(20, 20), max=2)


while animation:
    for i in sorted(os.listdir(path)):
        if count == 1:
            fname = i
            PS = Processing(path, fname)
            im = PS.read_image()
            gray, blur = PS.filter_and_threshold()

            ROI = PS.select_roi()
            mask = PS.mask_and_image(ROI)

            kp, dst, feature_image = PS.feature_detect()

            kp_list = []
            for i in range(len(kp)):
                kp_list.append(kp[i].pt)

            cv2.imshow('HEART VIDEO | FRAME: %s' % count, feature_image)

            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            cv2.destroyAllWindows()

            kp_array = np.asarray(kp_list, dtype=np.float32)

            while 1:

                cv2.namedWindow('HEART VIDEO | FRAME: %s' % count)
                cv2.setMouseCallback('HEART VIDEO | FRAME: %s' % count, CS.select_point)
                cv2.imshow('HEART VIDEO | FRAME: %s' % count, im)

                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    break
            cv2.destroyAllWindows()

            cs_array = np.asarray(CS.points, dtype=np.float32)

            p0 = np.concatenate((kp_array, cs_array))

        else:
            if PS is not None: del PS

            fname = i
            PS = Processing(path, fname)
            nxt_im = PS.read_image()
            nxt_gray, nxt_blur = PS.filter_and_threshold()

            p1, st, err = lk.lk(gray, nxt_gray, p0)

            for points in range(len(p1)):
                cv2.circle(nxt_im, (int(p1[points][0]), int(p1[points][1])), 4, (0, 255, 0), -1)
            cv2.imshow('HEART VIDEO', nxt_im)

            p0 = p1
            im, gray, blur = nxt_im, nxt_gray, nxt_blur

        count += 1
        time.sleep(0.1)
        k = cv2.waitKey(20) & 0xFF

        print("Frame : %s" % i)
        if k == 27:
            animation = False
            break

cv2.destroyAllWindows()
