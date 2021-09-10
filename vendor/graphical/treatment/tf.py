# Copyright (C) 2001-2007 Python Software Foundation
# Author: Ben
# Contact: email-sig@python.org

__all__ = [
    'Transform',
    ]

import numpy as np

class Transform(object):

    def __init__(self,img):
        # Read image
        self.img = img

    def imread(self):
        return self.img

    # function: BGR -> RGB
    def bgr2rgb(self):
        b = self.img[:, :, 0].copy()
        g = self.img[:, :, 1].copy()
        r = self.img[:, :, 2].copy()

        # RGB > BGR
        self.img[:, :, 0] = r
        self.img[:, :, 1] = g
        self.img[:, :, 2] = b

        return self.img

    # Gray scale
    def bgr2gray(self):
        b = self.img[:, :, 0].copy()
        g = self.img[:, :, 1].copy()
        r = self.img[:, :, 2].copy()

        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        return out

    def binarization(self, th=128):
        img = self.bgr2gray()
        img[img < th] = 0
        img[img >= th] = 255
        return img

    # Otsu Binarization
    def otsu_binarization(self, th=128):
        H, W, C = self.img.shape
        out = self.bgr2gray()
        max_sigma = 0
        max_t = 0
        # determine threshold
        for _t in range(1, 255):
            v0 = out[np.where(out < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            w0 = len(v0) / (H * W)
            v1 = out[np.where(out >= _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H * W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)
            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t

        # Binarization
        print("threshold >>", max_t)
        th = max_t
        out[out < th] = 0
        out[out >= th] = 255

        return out

    # BGR -> HSV
    def BGR2HSV(self):
        np.seterr(divide='ignore', invalid='ignore')
        img = self.img.copy() / 255.

        hsv = np.zeros_like(img, dtype=np.float32)

        # get max and min
        max_v = np.max(img, axis=2).copy()
        min_v = np.min(img, axis=2).copy()
        min_arg = np.argmin(img, axis=2)

        # H
        hsv[..., 0][np.where(max_v == min_v)] = 0
        ## if min == B
        ind = np.where(min_arg == 0)
        hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
        ## if min == R
        ind = np.where(min_arg == 2)
        hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
        ## if min == G
        ind = np.where(min_arg == 1)
        hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300

        # S
        hsv[..., 1] = max_v.copy() - min_v.copy()

        # V
        hsv[..., 2] = max_v.copy()

        return hsv

    def HSV2BGR(self):

        hsv = self.BGR2HSV()

        img = self.img.copy() / 255.

        # get max and min
        max_v = np.max(img, axis=2).copy()
        min_v = np.min(img, axis=2).copy()

        out = np.zeros_like(img)

        H = hsv[..., 0]
        S = hsv[..., 1]
        V = hsv[..., 2]

        C = S
        H_ = H / 60.
        X = C * (1 - np.abs(H_ % 2 - 1))
        Z = np.zeros_like(H)

        vals = [[Z, X, C], [Z, C, X], [X, C, Z], [C, X, Z], [C, Z, X], [X, Z, C]]

        for i in range(6):
            ind = np.where((i <= H_) & (H_ < (i + 1)))
            out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
            out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
            out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

        out[np.where(max_v == min_v)] = 0
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)

        return out

    # Transpose Hue
    def HSVTranspose(self):

        hsv = self.BGR2HSV()

        # Transpose Hue
        hsv[..., 0] = (hsv[..., 0] + 180) % 360
        # HSV > RGB
        out = self.HSV2BGR()

        return out