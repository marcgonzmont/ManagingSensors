import cv2
import numpy as np
import argparse

def nothing(x):
    pass

def computeThresh(img, thermal, alpha):
    max_t = thermal.max()
    min_t = thermal.min()
    mean_t = (max_t + min_t) // 2
    color = np.zeros_like(img)
    color[:, :, 2] = 255
    win_name = 'Thermal threshold'
    bar1 = 'Min (K)'
    bar2 = 'Max (K)'
    cv2.namedWindow(win_name)
    cv2.createTrackbar(bar1, win_name, min_t, mean_t, nothing)
    cv2.createTrackbar(bar2, win_name, mean_t, max_t, nothing)
    # result = img.copy()
    while (1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        min = cv2.getTrackbarPos(bar1, win_name)
        max = cv2.getTrackbarPos(bar2, win_name)

        mask = cv2.inRange(thermal, min, max, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), color)
        # mask = np.where(mask[:,:,:] == 255)
        result = cv2.addWeighted(img, alpha, mask, 1-alpha, 0)
        cv2.imshow(win_name, result)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
                    help="-i Color image")
    ap.add_argument("-t", "--thermal",
                    help="-t File with thermal information")
    args = vars(ap.parse_args())

    img_file = args["image"]
    thermal_file = args["thermal"]
    alpha = 0.5

    img = cv2.imread(img_file)
    img = cv2.resize(img, (240,320), interpolation= cv2.INTER_AREA)

    h, w = img.shape[:2]
    thermal_info = np.fromfile(thermal_file, dtype=np.float16)
    thermal_info = thermal_info // 100
    thermal_info = thermal_info.astype(int).reshape((320, 240))#.T
    # thermal_info = cv2.resize(thermal_info, (h, w), interpolation=cv2.INTER_LINEAR)
    computeThresh(img, thermal_info, alpha)