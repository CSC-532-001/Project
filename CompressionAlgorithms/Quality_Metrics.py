import numpy as np
import sewar.full_ref as metrics

class QualityMetrics:
    def __init__(self, original_img: np.ndarray, compressed_image: np.ndarray, is_greyscale = True):
        self.img = original_img
        self.img2 = compressed_image

        self.calculate_quality()

    def calculate_quality(self):
        self.mse = metrics.mse(self.img, self.img2)
        self.psnr = metrics.psnr(self.img, self.img2)
        self.psnrb = metrics.psnrb(self.img, self.img2)
        self.ssim = metrics.ssim(self.img, self.img2)
        self.mssim = metrics.msssim(self.img, self.img2)
        self.uqi = metrics.uqi(self.img, self.img2)
        self.ergas = metrics.ergas(self.img, self.img2)
        self.scc = metrics.scc(self.img, self.img2)
        self.sam = metrics.sam(self.img, self.img2)
        self.rase = metrics.rase(self.img, self.img2)
        self.vif = metrics.vifp(self.img, self.img2)

# Note: all of these are full-ref comparisons
