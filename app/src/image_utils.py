from typing import Dict, Any, Tuple
import numpy as np
import cv2
from astropy.io import fits

def to_fits_cube(img_bgr: np.ndarray) -> np.ndarray:
    """Converte BGR (H,W,3) para cubo FITS (3,H,W) em ordem de canais B,G,R."""
    return np.transpose(img_bgr, (2, 0, 1))


def from_fits_cube(cube: np.ndarray) -> np.ndarray:
    """Converte cubo FITS (3,H,W) em BGR (H,W,3)."""
    return np.transpose(cube, (1, 2, 0))


def equalize_img(img_bgr: np.ndarray) -> np.ndarray:
    """Equaliza histograma por canal (YCrCb) para preservar cores."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    out = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)


def calc_histogram(img_bgr: np.ndarray, bins: int = 32) -> np.ndarray:
    """Histograma concatenado por canal (BGR). Retorna vetor (bins*3,)."""
    hists = []
    for c in range(3):
        h = cv2.calcHist([img_bgr], [c], None, [bins], [0, 256])
        h = cv2.normalize(h, None).flatten()
        hists.append(h)
    return np.concatenate(hists)


def write_fits(path: str, img_bgr: np.ndarray, header: Dict[str, Any]) -> None:
    cube = to_fits_cube(img_bgr)
    hdu = fits.PrimaryHDU(cube)
    hdr = hdu.header
    for k, v in header.items():
        key = (k[:8]).upper()
        try:
            hdr[key] = v
        except Exception:
            hdr[key] = str(v)
    hdul = fits.HDUList([hdu])
    hdul.writeto(path, overwrite=True)


def read_fits(path: str) -> Tuple[np.ndarray, dict]:
    with fits.open(path) as hdul:
        cube = hdul[0].data
        header = dict(hdul[0].header)
    img_bgr = from_fits_cube(cube)
    return img_bgr, header