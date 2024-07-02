import numpy as np
import argparse

import cv2
import tqdm
import os


class Config:
    LEFT_PRIFIX = "left"
    RIGHT_PRIFIX = "right"
    PARAMETER_FILE = "stereo_parameter.npz"
    FOLDER = "images"


class CameraParameter:

    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist


class StereoCameraParameter:
    def __init__(self, left: CameraParameter, right: CameraParameter, R, T):
        self.left = left
        self.right = right
        self.R = R
        self.T = T


def StereoParamterFromFile(file: str):
    if not file.endswith(".npz"):
        raise ValueError("The file must be a .npz file")

    data = np.load(file)

    ret = data["ret"]
    mtx_left = data["mtx_left"]
    dist_left = data["dist_left"]
    mtx_right = data["mtx_right"]
    dist_right = data["dist_right"]
    R = data["R"]
    T = data["T"]
    E = data["E"]
    F = data["F"]
    rvecs_left = data["left_rvecs"]
    tvecs_left = data["left_tvecs"]
    rvect_right = data["right_rvecs"]
    tvect_right = data["right_tvecs"]

    return StereoCameraParameter(
        CameraParameter(mtx_left, dist_left),
        CameraParameter(mtx_right, dist_right),
        R,
        T,
    )


class DepthEstimation:

    def __init__(self):
        self.resolution = (1280, 720)
        pass

    def read_parameter(self, file: str):
        self.stereo_param = StereoParamterFromFile(file)

    def init_merge(self):
        rectify_left, rectify_right, proj_left, proj_right, Q, roi_left, roi_right = (
            cv2.stereoRectify(
                self.stereo_param.left.mtx,
                self.stereo_param.left.dist,
                self.stereo_param.right.mtx,
                self.stereo_param.right.dist,
                self.resolution,
                self.stereo_param.R,
                self.stereo_param.T,
            )
        )
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(
            self.stereo_param.left.mtx,
            self.stereo_param.left.dist,
            rectify_left,
            proj_left,
            self.resolution,
            cv2.CV_32FC1,
        )
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(
            self.stereo_param.right.mtx,
            self.stereo_param.right.dist,
            rectify_right,
            proj_right,
            self.resolution,
            cv2.CV_32FC1,
        )

        self.map_left_x = map_left_x
        self.map_left_y = map_left_y
        self.map_right_x = map_right_x
        self.map_right_y = map_right_y
        self.stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)
        self.Q = Q

    def read_image(self, file: str):
        image = cv2.imread(file, cv2.IMREAD_ANYDEPTH)
        if image.dtype == np.float32:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def process_pair(self, left: str, right: str):
        left = self.read_image(os.path.join(Config.FOLDER, left))
        right = self.read_image(os.path.join(Config.FOLDER, right))
        cv2.imwrite("left.png", left)
        print(left.shape, left.dtype, left.min(), left.max())

        left = cv2.remap(left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right = cv2.remap(right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        disparity = self.stereo.compute(left, right)
        depth = cv2.reprojectImageTo3D(disparity, self.Q)
        return depth

    def process_folder(self, folder: str):
        root_folder = os.path.join(folder)
        files = os.listdir(root_folder)
        files_left = [x for x in files if Config.LEFT_PRIFIX in x]
        files_right = [x for x in files if Config.RIGHT_PRIFIX in x]
        files_left.sort()
        files_right.sort()
        file_pairs = zip(files_left, files_right)

        for left, right in tqdm.tqdm(file_pairs):
            depth = self.process_pair(left, right)
            depth[depth == np.inf] = 0
            depth[depth < 0] = 0
            print(depth.shape, depth.dtype, depth.min(), depth.max())

            depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(folder, "depth_" + left + ".png"), depth)


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parameter_file",
        type=str,
        default="stereo_parameter.npz",
        help="The file containing the stereo camera parameters",
        required=True,
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="images",
        help="The folder containing the images to be processed",
        required=True,
    )

    parser.add_argument(
        "--prefix_left",
        type=str,
        default="left",
        help="The prefix for the left camera images",
    )
    parser.add_argument(
        "--prefix_right",
        type=str,
        default="right",
        help="The prefix for the right camera images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    depth_estimation = DepthEstimation()
    depth_estimation.read_parameter(args.parameter_file)

    Config.FOLDER = args.folder
    Config.LEFT_PRIFIX = args.prefix_left
    Config.RIGHT_PRIFIX = args.prefix_right

    depth_estimation.init_merge()
    depth_estimation.process_folder(args.folder)
