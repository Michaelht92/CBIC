#  __author__ = 'Michael'
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from log import Logger
from lbp import get_gabor_lbp

logger = Logger(logname='log1.txt', loglevel=1, logger="gaborlbp.py").getlog()


def gabor_features(image, u=2, v=2):
    features = []
    logger.info("get gabor imagefeatures for image: " + image)
    image = cv2.imread(image, 1)
    src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    src_f = np.array(src, dtype=np.float32)
    src_f /= 255.
    kernel_size = 21
    sig = 5                               # igma 带宽，取常数5
    gm = 1.0                              # gamma 空间纵横比，一般取1
    ps = 0.0                              # psi 相位，一般取0
    for i in range(0, u):
        for j in range(0, v):
            lm = i * 12 / (u - 1) + 9
            th = j * np.pi / v
            kernel = cv2.getGaborKernel((kernel_size, kernel_size),
                                       sig, th, lm, gm, ps)
            dest = cv2.filter2D(src_f, cv2.CV_32F, kernel)
            dest = dest * 10
            features.extend(get_gabor_lbp(dest))
    return features


if __name__ == "__main__":
   image_path = r"D:\ICBC\corel\693.jpg"
   print gabor_features(image_path, 2, 2)