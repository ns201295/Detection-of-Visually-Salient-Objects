import cv2
import numpy as np
from matplotlib import pyplot as plt
import pywt

class Saliency:
    def __init__(self, img, gauss_kernel=(5, 5)):

        self.gauss_kernel = gauss_kernel
        self.frame_orig = img

        # downsample image for processing
        self.small_shape = (64, 64)
        self.frame_small = cv2.resize(img, self.small_shape)

        # whether we need to do the math (True) or it has already
        # been done (False)
        self.need_saliency_map = True

    def get_saliency_map(self):
        """Returns a saliency map

            This method generates a saliency map for the image that was
            passed to the class constructor.

            :returns: grayscale saliency map
        """
        if self.need_saliency_map:
            # haven't calculated saliency map for this image yet
            num_channels = 1
            if len(self.frame_orig.shape) == 2:
                # single channel
                sal = self._get_channel_sal_magn(self.frame_small)
            else:
                # multiple channels: consider each channel independently
                sal = np.zeros_like(self.frame_small).astype(np.float32)
                for c in xrange(self.frame_small.shape[2]):
                    small = self.frame_small[:, :, c]
                    sal[:, :, c] = self._get_channel_sal_magn(small)

                # overall saliency: channel mean
                sal = np.mean(sal, 2)

            # postprocess: blur, square, and normalize
            if self.gauss_kernel is not None:
                sal = cv2.GaussianBlur(sal, self.gauss_kernel, sigmaX=8,
                                       sigmaY=0)
            sal = sal**2
            sal = np.float32(sal)/np.max(sal)

            # scale up
            sal = cv2.resize(sal, self.frame_orig.shape[1::-1])

            # store a copy so we do the work only once per frame
            self.saliencyMap = sal
            self.need_saliency_map = False

        return self.saliencyMap

    def _get_channel_sal_magn(self, channel):

        # do DWT and get log-spectrum
        img_wt = pywt.dwt(channel,'db2')
        magnitude, angle = cv2.cartToPolar(np.real(img_wt),np.imag(img_wt))

        # get log amplitude
        log_ampl = np.log10(magnitude.clip(min=1e-9))

        # blur log amplitude with avg filter
        log_ampl_blur = cv2.blur(log_ampl, (3, 3))

        # residual
        residual = np.exp(log_ampl - log_ampl_blur)

        # back to cartesian frequency domain
        real_part, imag_part = cv2.polarToCart(residual, angle)
        img_combined = pywt.idwt(real_part + 1j*imag_part,'db2s')
        magnitude, _ = cv2.cartToPolar(np.real(img_combined),np.imag(img_combined))


        return magnitude

    # def calc_magnitude_spectrum(self):
    #     # convert the frame to grayscale if necessary
    #     if len(self.frame_orig.shape) > 2:
    #         frame = cv2.cvtColor(self.frame_orig, cv2.COLOR_BGR2GRAY)
    #     else:
    #         frame = self.frame_orig

    #     # expand the image to an optimal size for dwt
    #     rows, cols = self.frame_orig.shape[:2]
    #     nrows = pywt.getOptimalDWTSize(rows)
    #     ncols = pywt.getOptimalDWTSize(cols)
    #     frame = cv2.copyMakeBorder(frame, 0, ncols-cols, 0, nrows-rows,
    #                                cv2.BORDER_CONSTANT, value=0)

    #     # do dwt and get log-spectrum
    #     img_wt = pywt.dwt(frame,'db2')
    #     spectrum = np.log10(np.abs(pywt.dwtshift(img_wt)))

    #     # return for plotting
    #     return 255*spectrum/np.max(spectrum)



    def get_proto_objects_map(self, use_otsu=True):
   
        saliency = self.get_saliency_map()

        
        _, img_objects = cv2.threshold(np.uint8(saliency*255), 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return img_objects
