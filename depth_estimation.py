import numpy as np
import argparse
import cv2
import tqdm
import os
import pandas as pd
import open3d as o3d
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__), "modules", "depth", "stereo_transformer_module"
    )
)
sys.path.append(
    os.path.join(os.path.dirname(__file__), "modules", "depth", "raft_stereo")
)

from modules.depth.sbm_cuda import SBMCuda
from modules.depth.vpi_disparity import VPIDisparity
from modules.depth.raft_stereo_wrapper import RaftStereoWrapper
from modules.depth.sgbm_cuda import SGBMCuda


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
        self.resolution = (1440, 1080)
        self.tonemap = cv2.createTonemap(2.2)
        # print(cv2.getBuildInformation())
        # self.vit = StereoTransformer()
        self.raft = RaftStereoWrapper()
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

        self.Q = Q
        # Initialize CUDA StereoBM or StereoSGBM
        window_size = 5  # Adjust as needed
        min_disparity = 16
        num_disparities = 128  # Must be divisible by 16
        block_size = 5  # Adjust as needed

        # Create the StereoBM object
        self.stereoBM = SBMCuda(SBMCuda.Config())
        self.vpi = VPIDisparity(VPIDisparity.Config())
        self.stereoSGBM = SGBMCuda(SGBMCuda.Config())

    def read_image(self, file: str):
        image = cv2.imread(file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if image.dtype == np.float32:
            pass
            # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            # image = image.astype(np.uint8)
        # if len(image.shape) == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(image.shape, image.dtype, image.min(), image.max())
        return image.astype(np.float32)

    def normalize_image(self, img):
        normalized = (img - img.min()) / (img.max() - img.min())

        # normalized = normalized * 65535

        return normalized

    def read_image_pair(self, left: str, right: str):
        left = self.read_image(left)
        right = self.read_image(right)
        # left = self.normalize_image(left)
        # right = self.normalize_image(right)

        left = cv2.remap(left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right = cv2.remap(right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)

        return left, right

    def compute_depth_from_disparity(self, disparity_map):
        print(
            "Disparity",
            disparity_map.min(),
            disparity_map.max(),
            np.median(disparity_map),
        )
        baseline = np.linalg.norm(self.stereo_param.T)
        fx = self.stereo_param.left.mtx[0, 0]
        depth = (
            baseline
            * fx
            / (
                disparity_map
                + np.abs(
                    self.stereo_param.left.mtx[0, 2] - self.stereo_param.right.mtx[0, 2]
                )
            )
        )
        depth_idx_inf = depth == np.inf
        depth[depth_idx_inf] = 0
        depth_max = np.max(depth)
        depth[depth_idx_inf] = depth_max
        depth[depth < 0] = 0
        depth[disparity_map <= 0] = -1

        return depth

    def process_SBM_cuda(self, left, right):
        bitswifts = 256
        disparity_map = self.process_SBM_cuda_bitswift(
            left, right, bitswifts
        ) + self.process_SBM_cuda_bitswift(left, right, 1)
        return disparity_map

    def process_SBM_cuda_bitswift(self, left, right, bitswift=256):
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

    def process_pair(self, left: str, right: str, model=None, use_gray=False):
        left, right = self.read_image_pair(left, right)
        if use_gray:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disparity_map = model(left, right)
        # disparity_map = np.clip(disparity_map, 0, 1440)
        depth = self.compute_depth_from_disparity(disparity_map)

        return disparity_map, depth

    def process_grouped_folder(self, folder: str):
        """
        folder/image_left/image_left.hdr ...
        """

        root_folder = os.path.join(folder, "images_origin")
        sub_folders = [
            x
            for x in os.listdir(root_folder)
            if os.path.isdir(os.path.join(root_folder, x))
        ]
        sub_folder_left = [x for x in sub_folders if Config.LEFT_PRIFIX in x][0]
        sub_folder_right = [x for x in sub_folders if Config.RIGHT_PRIFIX in x][0]

        images_left = os.listdir(os.path.join(root_folder, sub_folder_left))
        images_right = os.listdir(os.path.join(root_folder, sub_folder_right))
        images_left.sort()
        images_right.sort()
        image_pairs = zip(images_left, images_right)

        os.makedirs(os.path.join(root_folder, "depth"), exist_ok=True)
        os.makedirs(os.path.join(root_folder, "point_cloud"), exist_ok=True)

        for i, (left, right) in enumerate(tqdm.tqdm(image_pairs)):
            left_p = os.path.join(root_folder, sub_folder_left, left)
            right_p = os.path.join(root_folder, sub_folder_right, right)

            models = [
                # ("RAFT", self.raft, False),
                # ("SBM", self.stereoBM, True),
                # ("SGBM", self.stereoSGBM, True),
                # (
                #     "SBM_256",
                #     lambda x, y: self.process_SBM_cuda_bitswift(x, y, 256),
                #     True,
                # ),
                # ("SBM_1", lambda x, y: self.process_SBM_cuda_bitswift(x, y, 1), True),
                ("VPI", self.vpi, True),
            ]

            for name, model, use_gray in models:
                disparity, depth = self.process_pair(left_p, right_p, model, use_gray)
                output_path = os.path.join(
                    root_folder,
                    "depth",
                    left.replace(Config.LEFT_PRIFIX, f"_{name}_depth_"),
                )

                cv2.imwrite(
                    output_path,
                    depth,
                )
                depth[depth < 0] = 0
                depth_color = cv2.applyColorMap(
                    cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(
                        np.uint8
                    ),
                    cv2.COLORMAP_JET,
                )

                disparity[disparity < 0] = 0

                disparity_color = cv2.applyColorMap(
                    cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(
                        np.uint8
                    ),
                    cv2.COLORMAP_JET,
                )

                cv2.imwrite(
                    output_path.split(".")[0] + "_color.png",
                    depth_color,
                )
                cv2.imwrite(
                    output_path.split(".")[0] + "_disparity_color.png",
                    disparity_color,
                )

                points, color = self.generate_point_cloud(
                    self.read_image(left_p), depth
                )
                # self.show_point_cloud(points, color)

                cv2.imshow(f"{name}_depth", depth_color)
                cv2.imshow(f"{name}_disparity", disparity_color)

                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(points)
                color[color > color.mean() * 3] = color.mean() * 3
                color = color[:, ::-1]  # Convert BGR to RGB
                point_cloud.colors = o3d.utility.Vector3dVector(color / color.max())
                o3d.io.write_point_cloud(
                    f"{output_path.split('.')[0].replace('depth','point_cloud')}_{name}_point_cloud.ply",
                    point_cloud,
                )

            cv2.waitKey(0)

    def generate_point_cloud(self, left_image, depth_image):
        xx, yy = np.meshgrid(
            np.arange(left_image.shape[1]), np.arange(left_image.shape[0])
        )
        cx1 = self.stereo_param.left.mtx[0, 2]
        cy = self.stereo_param.left.mtx[1, 2]
        fx = self.stereo_param.left.mtx[0, 0]
        fy = self.stereo_param.left.mtx[1, 1]
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
        point_grid = (
            np.stack(((xx - cx1) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=0)
            * depth_image
        )
        mask = np.ones((left_image.shape[0], left_image.shape[1]), dtype=bool)

        # Remove flying points
        mask[1:][np.abs(depth_image[1:] - depth_image[:-1]) > 1] = False
        mask[:, 1:][np.abs(depth_image[:, 1:] - depth_image[:, :-1]) > 1] = False
        mask = mask & (depth_image > 0)

        points = point_grid.transpose(1, 2, 0)[mask]
        colors = left_image[mask].astype(np.float64)
        print(colors.min(), colors.max(), colors.mean())
        print("Point Cloud Generated : ", points.shape, colors.shape)
        return points, colors
        # points_with_color = np.concatenate((points, colors), axis=1)

        # sampled_points = points_with_color[
        #     np.random.choice(points_with_color.shape[0], 10000, replace=False)
        # ]
        # return sampled_points[:, :3], sampled_points[:, 3:]

    def show_point_cloud(self, points, colors, wait_for_key=False):

        points = pd.DataFrame(points, columns=["x", "y", "z"])
        points["red"] = colors[:, 2]
        points["green"] = colors[:, 1]
        points["blue"] = colors[:, 0]
        # cloud = PyntCloud(points)
        # cloud.plot()

    def process_folder(self, folder: str):
        root_folder = os.path.join(folder)
        files = os.listdir(root_folder)
        files_left = [
            os.path.join(root_folder, x) for x in files if Config.LEFT_PRIFIX in x
        ]
        files_right = [
            os.path.join(root_folder, x) for x in files if Config.RIGHT_PRIFIX in x
        ]
        files_left.sort()
        files_right.sort()
        file_pairs = zip(files_left, files_right)

        for left, right in tqdm.tqdm(file_pairs):
            depth = self.process_pair(left, right)
            depth[depth == np.inf] = 0
            depth[depth < 0] = 0

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
        "--use_grouped",
        action="store_true",
        help="If the images are grouped by preprocessing",
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
    if args.use_grouped:
        depth_estimation.process_grouped_folder(args.folder)
    else:
        depth_estimation.process_folder(args.folder)
