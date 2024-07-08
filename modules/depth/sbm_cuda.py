import cv2
import numpy as np


class SBMCuda:
    class Config:
        window_size = 5  # Adjust as needed
        min_disparity = 16
        num_disparities = 128  # Must be divisible by 16
        block_size = 5  # Adjust as needed
        disp12MaxDiff = 1
        uniquenessRatio = 10
        speckleWindowSize = 100
        speckleRange = 32

    def __init__(self, config: Config) -> None:
        self.config = config

        self.stereo = cv2.cuda.createStereoBM(
            numDisparities=config.num_disparities, blockSize=config.block_size
        )

        self.stereo.setMinDisparity(config.min_disparity)
        self.stereo.setNumDisparities(config.num_disparities)
        self.stereo.setDisp12MaxDiff(config.disp12MaxDiff)
        self.stereo.setUniquenessRatio(config.uniquenessRatio)
        self.stereo.setSpeckleWindowSize(config.speckleWindowSize)
        self.stereo.setSpeckleRange(config.speckleRange)

        # Compute the disparity map using CUDA
        self.stream = cv2.cuda.Stream()

    def __call__(self, left: np.ndarray, right: np.ndarray):
        return self.__process_SBM_cuda(left, right, bitswift=256)

    def __process_SBM_cuda(self, left, right, bitswift=256):
        left = cv2.normalize(left, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
        right = cv2.normalize(right, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)
        left = left / bitswift
        right = right / bitswift
        left[left > 255] = 0
        right[right > 255] = 0
        left = left.astype(np.uint8)
        right = right.astype(np.uint8)
        left = cv2.cuda_GpuMat(left)
        right = cv2.cuda_GpuMat(right)
        disparity = self.stereo.compute(left, right, self.stream)
        disparity = disparity.download()
        return disparity
