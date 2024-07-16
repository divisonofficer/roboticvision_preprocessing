import numpy as np


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
