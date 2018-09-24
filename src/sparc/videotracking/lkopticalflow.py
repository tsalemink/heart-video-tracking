import cv2


class LKOpticalFlow:
    def __init__(self, win=(20, 20), max_level=2):
        self.lk_params = dict(winSize=win, maxLevel=max_level,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def lk(self, curr_im, nxt_im, p0):
        p1, st, err = cv2.calcOpticalFlowPyrLK(curr_im, nxt_im, p0, None, **self.lk_params)

        return p1, st, err
