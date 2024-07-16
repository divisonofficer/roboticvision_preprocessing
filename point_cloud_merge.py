import argparse
import open3d as o3d
import cv2
import numpy as np
import os
import sqlite3
from typing import Optional
from modules.stereo.camera_parameter import StereoParamterFromFile
from modules.stereo.pose_calibration import PoseOptimizer
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares


def parseArgs():
    parser = argparse.ArgumentParser(description="Merge point clouds")
    parser.add_argument(
        "--folder", type=str, help="Folder containing point clouds", required=True
    )
    parser.add_argument(
        "--colmap_output", type=str, help="Colmap output folder", default="sparse/0"
    )
    parser.add_argument(
        "--database", type=str, help="Colmap database file", default="database.db"
    )
    parser.add_argument(
        "--voxel_size", type=float, help="Voxel size for downsampling", default=0.01
    )
    parser.add_argument(
        "--image_prefix",
        type=str,
        help="Image prefix",
        default="jai_1600_left_channel_0_fusion.png",
    )
    parser.add_argument(
        "--ply_infix", type=str, help="Infix for ply files", default="VPI"
    )
    return parser.parse_args()


COLOR_TABLES = [
    [0, 0, 0],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [128, 128, 255],
    [128, 0, 0],
    [0, 128, 0],
]

# Scale the color values to the range of 0 to 1.0
COLOR_TABLES = [[r / 255, g / 255, b / 255] for r, g, b in COLOR_TABLES]


class Config:
    Folder = ""
    ColampSparse = ""
    OUTPUT = "combined.ply"
    IMAGE_PREFIX = "jai_1600_left_channel_0_fusion.png"
    PLY_INFIX = "VPI"
    Voxel_Size = 0.05
    PLY_VALUE_SCALE = 0.001
    CAMERA_NPZ = "calibration.npz"


class PointCloudMerge:
    def __init__(self):
        pass

    def find_database_index(self, ply_filename):
        ply_filename = ply_filename.split("/")[-1]
        ply_filename = ply_filename.split("__")
        index_begin = 2
        while ply_filename[index_begin].isdigit():
            index_begin += 1
        ply_filename_index = "__".join(ply_filename[:index_begin])
        database_index = (
            f"{Config.IMAGE_PREFIX}/{ply_filename_index}__{Config.IMAGE_PREFIX}"
        )
        return database_index

    def read_database(self):
        self.conn = sqlite3.connect(Config.database)
        self.camera_calibration = StereoParamterFromFile(Config.CAMERA_NPZ)

    def db_read_image(self, image_name):
        query = f"SELECT * FROM images WHERE name = '{image_name}'"
        cursor = self.conn.execute(query)
        for row in cursor:
            return row
        return None

    def translation_points(
        self, points: np.ndarray, translation: np.ndarray, rotation: np.ndarray
    ):
        points = points @ rotation
        points = points + translation
        return points

    def process_ply(self, ply_file_name: str):
        ply = o3d.io.read_point_cloud(ply_file_name)

        # print(f"Original points: {len(ply.points)}")
        points = np.asarray(ply.points) * Config.PLY_VALUE_SCALE
        colors = np.asarray(ply.colors)
        distance_threshold = 2000 * Config.PLY_VALUE_SCALE
        colors = colors[np.linalg.norm(points, axis=1) <= distance_threshold]
        points = points[np.linalg.norm(points, axis=1) <= distance_threshold]
        ply.points = o3d.utility.Vector3dVector(points)
        ply.colors = o3d.utility.Vector3dVector(colors)
        ply = ply.voxel_down_sample(voxel_size=Config.Voxel_Size)
        points = np.asarray(ply.points)
        image_name = self.find_database_index(ply_file_name)
        # matrix34 = self.get_optimized_translation(image_name)
        matrix34 = self.get_transform_matrix(image_name)
        matrix34[0, 3] = matrix34[0, 3] * 1
        matrix34[2, 3] = matrix34[2, 3] * 1
        print(f"Translation: {matrix34[:3, 3]}")
        points = points @ matrix34[:3, :3]
        points -= matrix34[:3, 3] * self.translation_scale
        colors = np.asarray(ply.colors)
        return points, colors

    def merge_point_clouds(self, points_origin: np.ndarray, points_new: np.ndarray):
        points_merged = np.concatenate((points_origin, points_new), axis=0)
        points_merged, index = np.unique(points_merged, axis=0, return_index=True)
        return points_merged, index

    def poseprocess_points(
        self, points: np.ndarray, colors: Optional[np.ndarray] = None
    ):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
        cleaned, _ = point_cloud.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        return cleaned

    def get_translation_and_rotation(self, image_name):
        matrix = self.get_transform_matrix(image_name)
        translation = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        return translation, rotation_matrix

    def get_optimized_translation(self, image_name):
        image_indexes = [
            image_name,
            image_name.replace("channel_0", "channel_1"),
            image_name.replace("left", "right"),
            image_name.replace("left", "right").replace("channel_0", "channel_1"),
        ]

        image_matrix = [
            self.get_transform_matrix(image_index) for image_index in image_indexes
        ]
        image_matrix = {
            "A": image_matrix[0],
            "B": image_matrix[1],
            "C": image_matrix[2],
            "D": image_matrix[3],
        }
        stereo_matrix = np.hstack(
            (
                self.camera_calibration.R,
                self.camera_calibration.T
                * Config.PLY_VALUE_SCALE
                / self.translation_scale,
            )
        )
        optimizer = PoseOptimizer(image_matrix, stereo_matrix)
        optimizer.optimize()
        optimized_matrix = optimizer.get_optimized_poses()
        # print(f"Original:")
        # for key, value in image_matrix.items():
        #     print(f"{key}: {value[:,3]}")
        # print(f"Optimized:")
        # for key, value in optimized_matrix.items():
        #     print(f"{key}: {value[:,3]}")
        optimized_matrix["A"][:, 3] = (
            optimized_matrix["A"][:, 3] + optimized_matrix["C"][:, 3]
        ) / 2
        return optimized_matrix["A"]

    def get_transform_matrix(self, image_name):
        image_data = self.db_read_image(image_name)
        translation = (
            np.array([image_data[7], image_data[8], image_data[9]])
            * Config.PLY_VALUE_SCALE
        )
        # print(f"Translation: {translation}")
        rotation = np.array(image_data[3:7])
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(
            rotation
        )
        transformation_matrix = translation_matrix @ rotation_matrix

        return transformation_matrix

    def compute_camera_translation_scale(self):
        ply_folder = os.path.join(Config.Folder, "images_hdr", "point_cloud")
        ply_files = [
            f
            for f in os.listdir(ply_folder)
            if f.endswith(".ply") and Config.PLY_INFIX in f
        ]
        translation_diffs = []
        for ply_file in ply_files:
            image_name = self.find_database_index(ply_file)
            image_name_right = image_name.replace("left", "right")
            translation_left, _ = self.get_translation_and_rotation(image_name)
            translation_right, _ = self.get_translation_and_rotation(image_name_right)
            translation_diff = translation_right - translation_left
            translation_diffs.append(translation_diff)
        print(f"Translation diffs: {translation_diffs}")
        scale = (
            np.linalg.norm(self.camera_calibration.T)
            / np.linalg.norm(np.median(translation_diffs, axis=0))
            * Config.PLY_VALUE_SCALE
        )
        self.translation_scale = scale

    def process_folder(self):
        ply_folder = os.path.join(Config.Folder, "images_hdr", "point_cloud")
        ply_files = [
            f
            for f in os.listdir(ply_folder)
            if f.endswith(".ply") and Config.PLY_INFIX in f
        ]
        ply_files.sort()
        points_all = np.array([], dtype=np.float32).reshape(0, 3)
        colors_all = np.array([], dtype=np.float32).reshape(0, 3)
        for idx, ply in enumerate(ply_files):
            image_index = ply.split("/")[-1].split("__")[2:5]

            print(f"{idx} Processing {ply.split('/')[-1].split('__')[:5]}")
            points, colors = self.process_ply(os.path.join(ply_folder, ply))
            points_len = points.shape[0]
            colors = [
                COLOR_TABLES[int(image_index[1]) % len(COLOR_TABLES)]
                for _ in range(points_len)
            ]
            points_all, all_index = self.merge_point_clouds(points_all, points)
            colors_all = np.concatenate((colors_all, colors), axis=0)
            colors_all = colors_all[all_index]
            print(f"Points added: {points_len}, Total points: {points_all.shape[0]}")
            # if idx > 50:
            #     break

        points_all = self.poseprocess_points(points_all, colors_all)
        o3d.io.write_point_cloud(Config.OUTPUT, points_all)


if __name__ == "__main__":
    args = parseArgs()
    Config.Folder = args.folder
    Config.database = os.path.join(args.folder, args.database)
    Config.OUTPUT = os.path.join(Config.Folder, Config.OUTPUT)
    Config.Voxel_Size = args.voxel_size
    merger = PointCloudMerge()
    merger.read_database()
    merger.compute_camera_translation_scale()
    merger.process_folder()
    print(f"Point cloud merged to {Config.OUTPUT}")
