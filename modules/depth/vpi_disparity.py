import vpi
import numpy as np
import cv2


class VPIDisparity:
    class Config:
        scale = 1
        minDisparity = 16
        maxDisparity = 224
        includeDiagonals = True
        numPasses = 3
        calcConf = True
        downscale = 1
        windowSize = 23
        quality = 8
        backend = vpi.Backend.CUDA
        conftype = vpi.ConfidenceType.RELATIVE
        normalizeMax = 65535

    def __init__(self, config: Config) -> None:
        self.config = config

    def __preprocess(self, image: np.ndarray, stream):
        image = cv2.normalize(image, None, 0, self.config.normalizeMax, cv2.NORM_MINMAX)
        # Load input into a vpi.Image and convert it to grayscale, 16bpp
        with self.config.backend:
            with stream:
                image = vpi.asimage(image).convert(
                    vpi.Format.Y16_ER, scale=self.config.scale
                )
        return image

    def get_disparity_and_confidence(self, left: np.ndarray, right: np.ndarray):
        # pixel value scaling factor when loading input
        config = self.config
        # conftype = vpi.ConfidenceType.RELATIVE
        # Streams for left and right independent pre-processing
        streamLeft = vpi.Stream()
        streamRight = vpi.Stream()

        left = self.__preprocess(left, streamLeft)
        right = self.__preprocess(right, streamRight)

        confidenceU16 = None
        outWidth = (left.size[0] + config.downscale - 1) // config.downscale
        outHeight = (left.size[1] + config.downscale - 1) // config.downscale
        if config.calcConf:
            confidenceU16 = vpi.Image((outWidth, outHeight), vpi.Format.U16)

        # Estimate stereo disparity.
        disparityS16 = self.__stereodisp(left, right, confidenceU16, streamLeft)
        # Postprocess results and save them to disk
        with streamLeft, vpi.Backend.CUDA:
            # Some backends outputs disparities in block-linear format, we must convert them to
            # pitch-linear for consistency with other backends.
            if disparityS16.format == vpi.Format.S16_BL:
                disparityS16 = disparityS16.convert(
                    vpi.Format.S16, backend=vpi.Backend.VIC
                )

        return disparityS16.cpu(), confidenceU16.cpu()

    def get_disparity_multichannel(
        self,
        left_viz: np.ndarray,
        right_viz: np.ndarray,
        left_nir: np.ndarray,
        right_nir: np.ndarray,
    ):
        disparityViz, confidenceViz = self.get_disparity_and_confidence(
            left_viz, right_viz
        )
        disparityNir, confidenceNir = self.get_disparity_and_confidence(
            left_nir, right_nir
        )
        disparityViz = disparityViz.astype(np.float32)
        disparityNir = disparityNir.astype(np.float32)
        confidenceViz = confidenceViz.astype(np.float32) / confidenceViz.max()
        confidenceNir = confidenceNir.astype(np.float32) / confidenceNir.max()
        disparityViz[disparityViz < 0] = 0
        disparityNir[disparityNir < 0] = 0
        disparityViz[confidenceViz < confidenceNir] = 0
        disparityNir[confidenceNir < confidenceViz] = 0
        disparityNir[confidenceNir < 0.8] = 0
        disparityViz[confidenceViz < 0.8] = 0

        disparity = disparityViz + disparityNir
        disparity = disparity * 255.0 / (32 * self.config.maxDisparity)
        disparity = disparity.astype(np.uint8)

        confidence_mask = cv2.bitwise_and(
            cv2.bitwise_and(
                (confidenceViz < 0.8).astype(np.uint8),
                (confidenceViz > 0).astype(np.uint8),
            ),
            cv2.bitwise_and(
                (confidenceNir < 0.8).astype(np.uint8),
                (confidenceNir > 0).astype(np.uint8),
            ),
        )

        disparity = self.__interpolation(disparity, confidence_mask)

        return disparity

    def __call__(self, left: np.ndarray, right: np.ndarray):

        disparityS16, confidence = self.get_disparity_and_confidence(left, right)
        # disparityColor = self.__apply_color_map(disparityS16, confidence)
        confidence = confidence.astype(np.float32) / confidence.max()
        disparity_numpy = (
            disparityS16.astype(np.float32) * 255.0 / (32 * self.config.maxDisparity)
        ).astype(np.uint8)

        disparity_numpy[confidence < 0.8] = 0
        confidence_mask = (
            cv2.bitwise_and(
                (confidence < 0.8).astype(np.uint8),
                (confidence > 0.00).astype(np.uint8),
            )
            * 255
        )
        disparity_numpy = self.__interpolation(disparity_numpy, confidence_mask)

        return disparity_numpy

    def __interpolation(self, disparity, mask):
        borderSize = 6
        disparity = cv2.copyMakeBorder(
            disparity,
            borderSize,
            borderSize,
            borderSize,
            borderSize,
            cv2.BORDER_REPLICATE,
        )  # Convert mask to uint8
        mask = cv2.copyMakeBorder(
            mask, borderSize, borderSize, borderSize, borderSize, cv2.BORDER_REPLICATE
        )
        disparity = cv2.inpaint(disparity, mask, 5, cv2.INPAINT_TELEA)
        return disparity[borderSize:-borderSize, borderSize:-borderSize]

    def __stereodisp(self, left, right, confidenceU16, streamStereo):
        config = self.config
        with streamStereo, config.backend:
            disparityS16 = vpi.stereodisp(
                left,
                right,
                downscale=config.downscale,
                out_confmap=confidenceU16,
                window=config.windowSize,
                maxdisp=config.maxDisparity,
                confthreshold=22222,
                quality=config.quality,
                conftype=config.conftype,
                mindisp=config.minDisparity,
                p1=4,
                p2=128,
                p2alpha=1,
                uniqueness=0.95,
                includediagonals=config.includeDiagonals,
                numpasses=config.numPasses,
            )
        return disparityS16

    def __apply_color_map(self, disparityS16, confidenceU16):
        # Scale disparity and confidence map so that values like between 0 and 255.
        # Disparities are in Q10.5 format, so to map it to float, it gets
        # divided by 32. Then the resulting disparity range, from 0 to
        # stereo.maxDisparity gets mapped to 0-255 for proper output.
        # Copy disparity values back to the CPU.
        disparityU8 = disparityS16.convert(
            vpi.Format.U8, scale=255.0 / (32 * self.config.maxDisparity)
        ).cpu()

        # Apply JET colormap to turn the disparities into color, reddish hues
        # represent objects closer to the camera, blueish are farther away.
        disparityColor = cv2.applyColorMap(disparityU8, cv2.COLORMAP_JET)

        # Converts to RGB for output with PIL.
        disparityColor = cv2.cvtColor(disparityColor, cv2.COLOR_BGR2RGB)

        if confidenceU16 is not None:
            confidenceU8 = confidenceU16.convert(
                vpi.Format.U8, scale=255.0 / 65535
            ).cpu()

            # When pixel confidence is 0, its color in the disparity is black.
            mask = cv2.threshold(confidenceU8, 1, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            disparityColor = cv2.bitwise_and(disparityColor, mask)
        return disparityColor

    def __runtime_imshow(self, image_dict):
        for key, value in image_dict.items():
            cv2.imshow(key, value)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
