import os
import sqlite3
import subprocess
import threading
import json
from typing import Optional, Callable
from queue import Queue, Empty
import sys
import time
import struct

ON_POSIX = "posix" in sys.builtin_module_names


class ColmapWrap:
    SPACE_FOLDER = ""

    def __init__(self, tqdm_callback: Callable[[str, float], None]):
        self.camera_params: Optional[tuple[list, list]] = None
        self.tqdm_callback = tqdm_callback
        self.camera_params = None
        pass

    def set_space_folder(self, folder):
        self.SPACE_FOLDER = folder
        self.read_camera_meta()

    def find_db_file(self):
        db_file = os.path.join(self.SPACE_FOLDER, "database.db")
        if os.path.exists(db_file):
            return db_file
        else:
            return None

    def read_camera_meta(self):
        if not os.path.exists(os.path.join(self.SPACE_FOLDER, "camera_meta.json")):
            self.camera_params = None
        camera_meta_dict = json.load(
            open(os.path.join(self.SPACE_FOLDER, "camera_meta.json"))
        )

        self.camera_params: tuple[list, list] = (
            camera_meta_dict["colmap_left"],
            camera_meta_dict["colmap_right"],
        )

    def create_mysql_db(self):
        db_file = self.find_db_file()
        if db_file is None:
            db_file = os.path.join(self.SPACE_FOLDER, "database.db")
            subprocess.run(["colmap", "database_creator", "--database_path", db_file])

            return self.find_db_file()
        else:
            print("Database already exists")
            return db_file

    def update_camera_parameter(self):
        sql = sqlite3.connect(self.create_mysql_db())
        cursor = sql.cursor()
        cameras = cursor.execute("SELECT camera_id, params FROM cameras").fetchall()
        param_left_byte_encoded = struct.pack(
            f"<{len(self.camera_params[0])}d", *self.camera_params[0]
        )
        param_right_byte_encoded = struct.pack(
            f"<{len(self.camera_params[1])}d", *self.camera_params[1]
        )

        print(self.camera_params)

        for camera_id in cameras:
            print(camera_id[0], camera_id[0] % 2)
            if camera_id[0] % 2 == 1:
                cursor.execute(
                    "UPDATE cameras SET params = ? WHERE camera_id = ?",
                    (param_left_byte_encoded, camera_id[0]),
                )
            else:
                cursor.execute(
                    "UPDATE cameras SET params = ? WHERE camera_id = ?",
                    (param_right_byte_encoded, camera_id[0]),
                )
        sql.commit()

    def run_feature_matcher(self):

        if self.camera_params is None:
            print("Camera parameter is not set")
            return

        self.update_camera_parameter()

        def cmd_feature_matcher_callback(output):
            if "Matching block" in output:
                progress_str = (
                    output.split("Matching block ")[1].split("]")[0][1:].split(", ")
                )
                progress_str = [x.split("/") for x in progress_str]
                total = int(progress_str[1][1]) ** 2
                progress = (
                    (int(progress_str[0][0]) - 1) * int(progress_str[1][1])
                    + int(progress_str[1][0])
                    - 1
                )
                self.tqdm_callback(
                    f"Feature Match {progress}/{total}", float(progress) / float(total)
                )

        threading.Thread(
            target=self.run_cmd,
            args=(
                [
                    "colmap",
                    "exhaustive_matcher",
                    "--database_path",
                    self.create_mysql_db(),
                ],
                cmd_feature_matcher_callback,
            ),
        ).start()

    def run_feature_extractor(self):

        if self.camera_params is None:
            print("Camera parameter is not set")
            return
        image_path = os.path.join(self.SPACE_FOLDER, "images_origin")

        def cmd_feature_extractor_callback(output):
            if "Processed file" in output:
                progress_str = output.split("Processed file ")[1].split("]")[0][1:]
                progress, total = progress_str.split("/")
                self.tqdm_callback(
                    f"Feature Extract {progress_str}", float(progress) / float(total)
                )

        thread = threading.Thread(
            target=self.run_cmd,
            args=(
                [
                    "colmap",
                    "feature_extractor",
                    "--database_path",
                    self.create_mysql_db(),
                    "--image_path",
                    image_path,
                    "--ImageReader.single_camera_per_folder",
                    "1",
                    "--ImageReader.camera_model",
                    "OPENCV",
                    "--ImageReader.camera_params",
                    ",".join([str(x) for x in self.camera_params[0]]),
                ],
                cmd_feature_extractor_callback,
            ),
        )
        thread.start()
        return thread

    def run_model_converter(self):
        for model in os.listdir(os.path.join(self.SPACE_FOLDER, "sparse")):
            if model.isdigit():
                model_path = os.path.join(self.SPACE_FOLDER, "sparse", model)

                self.run_cmd(
                    [
                        "colmap",
                        "model_converter",
                        "--input_path",
                        model_path,
                        "--output_path",
                        model_path,
                        "--output_type",
                        "TXT",
                    ],
                    lambda x: print(x),
                )

    def run_model_convert_thread(self):
        def run():
            self.run_model_converter()
            self.run_model_db_update()

        thread = threading.Thread(
            target=run,
        )
        thread.start()
        return thread

    def run_model_db_update(self):
        db = sqlite3.connect(self.create_mysql_db())

        for model in os.listdir(os.path.join(self.SPACE_FOLDER, "sparse")):
            if not model.isdigit():
                continue
            model_image_path = os.path.join(
                self.SPACE_FOLDER, "sparse", model, "images.txt"
            )
            images = []
            with open(model_image_path, "r") as f:
                images = f.readlines()
                images = [x.split(" ") for x in images if not x.strip().startswith("#")]
                images = [(images[x], images[x + 1]) for x in range(0, len(images), 2)]
            for idx, (image, points) in enumerate(images):
                self.tqdm_callback(
                    f"Updating {image[0]}", float(idx) / float(len(images))
                )
                image_id = int(image[0])
                qw = float(image[1])
                qx = float(image[2])
                qy = float(image[3])
                qz = float(image[4])
                tx = float(image[5])
                ty = float(image[6])
                tz = float(image[7])
                camera_id = int(image[8])
                name = image[9].strip()

                db.execute(
                    """
                        UPDATE images
                        SET name = ?, 
                            camera_id = ?, 
                            prior_qw = ?, 
                            prior_qx = ?, 
                            prior_qy = ?, 
                            prior_qz = ?, 
                            prior_tx = ?, 
                            prior_ty = ?, 
                            prior_tz = ?
                        WHERE image_id = ?
                        """,
                    (name, camera_id, qw, qx, qy, qz, tx, ty, tz, image_id),
                )

        db.commit()

    def run_feature_mapper(self):
        if self.camera_params is None:
            print("Camera parameter is not set")
            return
        seed = str(int(time.time()))
        self.image_count = 0

        def callback_mapper(output: str):
            if "connected" in output:
                self.image_count = int(
                    output.split("connected")[1].split(")")[0].strip()
                )
            if "Registering image" in output:
                image_number = int(
                    output.split("Registering image ")[1]
                    .split("(")[1]
                    .replace(")", "")
                    .strip()
                )
                self.tqdm_callback(
                    f"Mapping {image_number}/{self.image_count}",
                    float(image_number) / float(self.image_count),
                )

        os.makedirs(os.path.join(self.SPACE_FOLDER, "sparse"), exist_ok=True)

        def run():
            self.run_cmd(
                [
                    "colmap",
                    "mapper",
                    "--database_path",
                    self.create_mysql_db(),
                    "--image_path",
                    os.path.join(self.SPACE_FOLDER, "images_origin"),
                    "--output_path",
                    os.path.join(self.SPACE_FOLDER, "sparse"),
                    "--random_seed",
                    seed,
                ],
                callback_mapper,
            )

            self.run_model_converter()

        thread = threading.Thread(
            target=run,
        )
        thread.start()
        return thread

    def run_fast_mapper(self):
        def run():
            self.run_feature_extractor().join()
            self.run_feature_mapper().join()
            self.run_feature_mapper().join()
            self.run_model_convert_thread().join()

        thread = threading.Thread(
            target=run,
        )
        thread.start()

    def run_cmd(self, CMD, callback: Callable[[str], None]):
        process = subprocess.Popen(
            CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            close_fds=True,
        )

        def queue_callback(output, _q):
            try:
                for line in iter(output.readline, b""):
                    _q.put(line)
                output.close()
            except Exception as e:
                print(f"Error in callback: {e}")

        q = Queue()
        t = threading.Thread(
            target=queue_callback, args=(process.stdout, q), daemon=True
        )
        t_e = threading.Thread(
            target=queue_callback, args=(process.stderr, q), daemon=True
        )
        t_e.start()
        t.start()

        while True:
            try:
                output = q.get_nowait()
            except Empty:
                time.sleep(0.1)
                pass
            else:
                callback(output)
            if process.poll() is not None:
                # 프로세스가 종료되었으면 루프를 탈출합니다.
                break
