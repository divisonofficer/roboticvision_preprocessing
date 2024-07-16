import numpy as np
from scipy.optimize import least_squares


class PoseOptimizer:
    def __init__(self, initial_poses, stereo_pose):
        """
        Initialize the PoseOptimizer with the initial absolute poses and the relative poses of B, C, D with respect to A.

        Parameters:
        - initial_poses: dict with keys 'A', 'B', 'C', 'D' and values as 4x4 transformation matrices
                         representing the initial absolute poses.
        - stereo_pose : matrix of result of stereo calibration
        """
        self.initial_poses = initial_poses
        self.relative_poses = {
            "B": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            "C": stereo_pose,
            "D": stereo_pose,
        }
        self.optimized_poses = None

    def matrix_to_params(self, matrix):
        """Convert a 4x4 transformation matrix to a 12-element parameter vector."""
        return matrix[:3, :].flatten()

    def params_to_matrix(self, params):
        """Convert a 12-element parameter vector to a 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, :] = params.reshape(3, 4)
        return matrix

    def relative_pose_error(self, params):
        """
        Compute the error between the given relative poses and the poses derived from the optimization parameters.

        Parameters:
        - params: 48-element array representing the 4x4 transformation matrices of A, B, C, and D.

        Returns:
        - errors: concatenated translation and rotation errors for all relative poses.
        """
        A_matrix = self.params_to_matrix(params[0:12])
        B_matrix = self.params_to_matrix(params[12:24])
        C_matrix = self.params_to_matrix(params[24:36])
        D_matrix = self.params_to_matrix(params[36:48])

        abs_matrices = {"A": A_matrix, "B": B_matrix, "C": C_matrix, "D": D_matrix}

        errors = []

        for pair, rel_matrix in self.relative_poses.items():
            cam1 = "A"
            cam2 = pair
            predicted_matrix = np.linalg.inv(abs_matrices[cam1]) @ abs_matrices[cam2]
            translation_error = rel_matrix[:3, 3] - predicted_matrix[:3, 3]
            rotation_error = rel_matrix[:3, :3] - predicted_matrix[:3, :3]
            errors.append(
                np.concatenate([translation_error.flatten(), rotation_error.flatten()])
            )

        return np.concatenate(errors)

    def optimize(self):
        """Perform the optimization to find the best absolute poses for A, B, C, D."""
        initial_guess = np.concatenate(
            [
                self.matrix_to_params(self.initial_poses["A"]),
                self.matrix_to_params(self.initial_poses["B"]),
                self.matrix_to_params(self.initial_poses["C"]),
                self.matrix_to_params(self.initial_poses["D"]),
            ]
        )

        result = least_squares(self.relative_pose_error, initial_guess)

        self.optimized_poses = {
            "A": self.params_to_matrix(result.x[0:12]),
            "B": self.params_to_matrix(result.x[12:24]),
            "C": self.params_to_matrix(result.x[24:36]),
            "D": self.params_to_matrix(result.x[36:48]),
        }

    def get_optimized_poses(self):
        """Return the optimized absolute poses."""
        if self.optimized_poses is None:
            raise ValueError(
                "Optimization has not been run yet. Call the 'optimize' method first."
            )
        return self.optimized_poses
