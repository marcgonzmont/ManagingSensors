import cv2
import numpy as np
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image",
                    help="-i Color image")
    ap.add_argument("-t", "--thermal",
                    help="-t File with thermal information")
    args = vars(ap.parse_args())

    img_file = args["image"]
    thermal_file = args["thermal"]

    img = cv2.imread(img_file)

    thermal_info = np.fromfile(thermal_file, dtype=np.float16)
    thermal_info = thermal_info // 100
    thermal_info = thermal_info.astype(int).reshape((320, 240))#.T
    max_t = thermal_info.max()
    min_t = thermal_info.min()
    print(max_t, min_t)
    # cv2.namedWindow('Thermal segmentation')
