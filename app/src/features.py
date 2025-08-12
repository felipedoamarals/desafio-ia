from typing import Dict
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

def color_hist_features(img_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    from image_utils import calc_histogram
    return calc_histogram(img_bgr, bins=bins)


def texture_glcm_features(img_gray: np.ndarray) -> np.ndarray:
    """Extrai contraste/energia/homogeneidade via GLCM."""
    img8 = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    glcm = graycomatrix(img8, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ("contrast", "energy", "homogeneity"):
        feats.append(graycoprops(glcm, prop).ravel())
    return np.concatenate(feats)


def build_feature_vector(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    f_color = color_hist_features(img_bgr, bins=32)
    f_tex = texture_glcm_features(gray)
    return np.concatenate([f_color, f_tex])