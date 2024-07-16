import os
from tqdm import tqdm
import cv2
import numpy as np
from fusion import fusion
import argparse
from typing import Optional, Callable
from ui.util.tqdm_list import TQDMList


_IMAGE_FILES = [
    "jai_1600_left_channel_0.png",
    "jai_1600_left_channel_1.png",
    "jai_1600_right_channel_0.png",
    "jai_1600_right_channel_1.png",
    # "oakd_left",
    # "oakd_right",
    # "oakd_rgb",
]

TQDM = tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        default="lab0624",
        help="The folder containing the images to be processed",
        required=True,
    )
    parser.add_argument(
        "--dest_folder",
        type=str,
        default="images_origin",
        help="The destination folder for the processed images",
    )
    parser.add_argument(
        "--rgb_nir_fusion",
        type=bool,
        default=False,
        help="Enable RGB-NIR fusion",
    )
    parser.add_argument(
        "--hdr_fusion",
        type=bool,
        default=False,
        help="Enable HDR fusion",
    )
    parser.add_argument("--use_mask", type=bool, default=False, help="Use mask")
    parser.add_argument(
        "--image_files",
        type=str,
        default=",".join(_IMAGE_FILES),
        help="The image files to be processed",
    )
    return parser.parse_args()


import subprocess


class Preprocessing:
    class HyperParamter:
        FOLDER = "lab0624"
        DEST_FOLDER = "images_origin"
        RGB_NIR_FUSION_ENABLED = False
        USE_MASK = False
        IMAGE_FILES = _IMAGE_FILES
        hdr_fusion = False

    def __init__(
        self, hyperparameter: Optional[HyperParamter], tqdm: Optional[Callable] = None
    ):
        if hyperparameter is not None:
            self.parameter = hyperparameter
        else:
            self.parameter = Preprocessing.HyperParamter()
            global TQDM
            TQDM = lambda x, desc=None: TQDMList(x, desc, tqdm)
            self.tqdm_callback = tqdm

    def load_gzip(self, file_path: str):
        space_id = file_path.split("/")[-1].split(".")[0]
        dest_folder = os.path.join("data", space_id)
        os.makedirs(dest_folder, exist_ok=True)
        subprocess.run(
            ["tar", "-xvf", file_path, "-C", dest_folder], stdout=subprocess.PIPE
        )
        space_folder = dest_folder
        if "tmp" in os.listdir(space_folder):
            space_folder = os.path.join(space_folder, "tmp")
        if "oakd_capture" in os.listdir(space_folder):
            space_folder = os.path.join(space_folder, "oakd_capture")
            space_folder = os.path.join(space_folder, os.listdir(space_folder)[0])
        for capture_folder in os.listdir(space_folder):
            capture_folder = os.path.join(space_folder, capture_folder)
            subprocess.run(["mv", capture_folder, dest_folder], stdout=subprocess.PIPE)
        self.parameter.FOLDER = dest_folder

        return dest_folder

    def examine_folder(self):
        key_found: dict[str, int] = {}
        capture_cnt = 0
        scene_cnt = 0
        for capture_folders in TQDM(os.listdir(self.parameter.FOLDER)):
            if not capture_folders.isdigit():
                continue
            if not os.path.isdir(os.path.join(self.parameter.FOLDER, capture_folders)):
                continue
            capture_cnt += 1
            for scene_folder in os.listdir(
                os.path.join(self.parameter.FOLDER, capture_folders)
            ):
                if not scene_folder.isdigit() or not os.path.isdir(
                    os.path.join(self.parameter.FOLDER, capture_folders, scene_folder)
                ):
                    continue
                scene_cnt += 1
                for image in os.listdir(
                    os.path.join(self.parameter.FOLDER, capture_folders, scene_folder)
                ):
                    if not image.endswith(".png") and not image.endswith(".hdr"):
                        continue
                    key = image.split(".png")[0]
                    if key not in key_found:
                        key_found[key] = 0
                    key_found[key] += 1
        return {
            "capture_cnt": capture_cnt,
            "scene_cnt": scene_cnt,
            "key_found": key_found,
        }

    def rgb_nir_fusion(
        rgb_image: cv2.typing.MatLike,
        nir_image: cv2.typing.MatLike,
        mask: cv2.typing.MatLike,
    ):
        return fusion(rgb_image, nir_image.mean(axis=2).astype(np.uint8), mask)

    def get_mask(self, folder: str):
        config = self.parameter
        image_list = [
            f"{x}/{y}"
            for x in os.listdir(folder)
            for y in os.listdir(os.path.join(folder, x))
            if y.endswith(".png")
        ]

        mask = 255 - np.zeros_like(
            cv2.imread(os.path.join(folder, image_list[0])), dtype=np.uint8
        )
        for image_name in image_list[:30]:
            print(image_name)
            image_path = os.path.join(folder, image_name)
            image = cv2.imread(image_path)
            image = 255 - cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 5, 11)
            mask = cv2.bitwise_and(mask, image)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(config.FOLDER, "mask.png"), mask)

    def refine_mask(
        self,
    ):
        config = self.parameter
        mask = cv2.imread(os.path.join(config.FOLDER, "mask.png"), cv2.IMREAD_GRAYSCALE)
        mask = 255 - mask
        mask[mask < 203] = 0
        mask[mask > 203] = 255
        cv2.imwrite(os.path.join(config.FOLDER, "mask_enhanced.png"), mask)

    def hdr_prepare_exposure_list(self, image_file_path: str):
        image_file_name = image_file_path.split("/")[-1].split(".")[0]
        image_folder = image_file_path.split(image_file_name)[0]
        image_path_list = [
            x
            for x in os.listdir(image_folder)
            if x.endswith(".png") and image_file_name in x
        ]
        image_path_list = [
            x for x in image_path_list if x.split(image_file_name)[1] != ".png"
        ]
        image_path_list.sort()
        exposure_times = [
            x.split(image_file_name + "_")[1].split(".png")[0] for x in image_path_list
        ]
        exposure_times = [int(x) for x in exposure_times if x.isdigit()]
        if len(exposure_times) < 4:
            return None
        exposure_times.sort()
        image_path_list = [
            os.path.join(image_folder, image_file_name + "_" + str(x) + ".png")
            for x in exposure_times
        ]
        exposure_times = (
            np.array(
                exposure_times,
                dtype=np.float32,
            )
            / 1000000.0
        )

        return image_path_list, exposure_times

    def hdr_fusion(self, image_file_path, image_path_list, exposure_times, response):
        image_anydepth = [
            cv2.imread(x, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            for x in image_path_list
        ]
        if any(x is None for x in image_anydepth):
            return

        images = [
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for x in image_anydepth
        ]
        if len(images[0].shape) == 2:
            images = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in images]
        hdr = self.merge.process(images, exposure_times, response)

        if len(hdr.shape) == 2:
            hdr = cv2.cvtColor(hdr, cv2.COLOR_GRAY2BGR)

        ldr = self.tonemap.process(hdr)

        fusion = self.mertge_mertges.process(images)
        image_file_path = image_file_path.replace(".png", "")
        cv2.imwrite(image_file_path + "_sdr.png", ldr * 255)
        cv2.imwrite(image_file_path + "_hdr.hdr", hdr)
        cv2.imwrite(image_file_path + "_fusion.png", fusion * 255)

        for i in images:
            del i
        for i in image_anydepth:
            del i

    def hdr_fusion_scene(self, scene_folder: str):
        images = os.listdir(scene_folder)
        image_prefix = {}
        for image in images:
            if not image.endswith(".png"):
                continue
            image = image.split(".png")[0]
            exposure_time = image.split("_")[-1]
            image = image.replace(f"_{exposure_time}", "")
            if not exposure_time.isdigit():
                continue
            if int(exposure_time) < 100:
                continue
            image_prefix[image] = os.path.join(scene_folder, image)

        return image_prefix

    def listdir_space_scene(self, folder: str):
        scene_folder_list: list[str] = []
        for capture_folders in os.listdir(folder):
            if not capture_folders.isdigit():
                continue
            if not os.path.isdir(os.path.join(folder, capture_folders)):
                continue
            for scene_folders in os.listdir(os.path.join(folder, capture_folders)):
                if not scene_folders.isdigit():
                    continue
                if not os.path.isdir(
                    os.path.join(folder, capture_folders, scene_folders)
                ):
                    continue
                scene_folder_list.append(
                    os.path.join(folder, capture_folders, scene_folders)
                )
        return scene_folder_list

    def hdr_fusion_module_init(self):
        self.merge = cv2.createMergeRobertson()
        self.calibrate = cv2.createCalibrateRobertson()
        self.mertge_mertges = cv2.createMergeMertens()
        self.tonemap = cv2.createTonemap(2.2)

    def hdr_fusion_space(self):

        config = self.parameter
        image_dict: dict[str, list[dict]] = {}
        for scene_folders in TQDM(
            self.listdir_space_scene(config.FOLDER), desc="HDR reading datas "
        ):

            image_prefix_dict = self.hdr_fusion_scene(scene_folders)
            for prefix, path in image_prefix_dict.items():
                if prefix not in image_dict:
                    image_dict[prefix] = []
                images, exposure_times = self.hdr_prepare_exposure_list(path)
                image_dict[prefix].append(
                    {
                        "path": path,
                        "images": images,
                        "exposure_times": exposure_times,
                    }
                )
        response: list[Optional[np.ndarray]] = [None, None]
        for prefix, image_list in TQDM(image_dict.items(), desc="HDR fusion "):
            channel = int(prefix.split("_")[-1])
            print(prefix)
            if response[channel] is None:
                self.hdr_fusion_module_init()
                images_all = []
                exposures_all = []
                for meta in image_list:
                    images_all.extend(meta["images"])
                    exposures_all.extend(meta["exposure_times"])
                    if len(images_all) > 64:
                        break
                print(exposures_all)
                response[channel] = self.computing_calibrate_response(
                    images_all, exposures_all
                )

            for meta in TQDM(image_list, desc=f"HDR fusion on {prefix}"):
                self.hdr_fusion(
                    meta["path"],
                    meta["images"],
                    meta["exposure_times"],
                    response[channel],
                )

    def computing_calibrate_response(
        self, image_files: list[str], exposure_times: list[float]
    ):
        images = [
            cv2.imread(x, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            for x in image_files
        ]
        images = [
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for x in images
        ]

        if len(images[0].shape) == 2:
            images = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in images]

        exposure_times = np.array(exposure_times, dtype=np.float32)
        self.tqdm_callback("Calibrating", 0)
        print("Calibrating")
        response = self.calibrate.process(images, exposure_times)
        print("Done")
        self.tqdm_callback("Calibrating Done", 1)
        for i in images:
            del i
        return response

    def group_images_full(self):
        config = self.parameter
        old_dest = config.DEST_FOLDER
        old_images = config.IMAGE_FILES
        config.DEST_FOLDER = "images_origin"
        config.IMAGE_FILES = [
            "jai_1600_left_channel_0_fusion.png",
            "jai_1600_left_channel_1_fusion.png",
            "jai_1600_right_channel_0_fusion.png",
            "jai_1600_right_channel_1_fusion.png",
        ]
        self.group_images()
        config.DEST_FOLDER = "images_hdr"
        config.IMAGE_FILES = [
            "jai_1600_left_channel_0_hdr.hdr",
            "jai_1600_left_channel_1_hdr.hdr",
            "jai_1600_right_channel_0_hdr.hdr",
            "jai_1600_right_channel_1_hdr.hdr",
        ]
        self.group_images()
        config.DEST_FOLDER = old_dest
        config.IMAGE_FILES = old_images

    def group_images(
        self,
    ):
        if self.parameter.USE_MASK:
            mask = cv2.imread(os.path.join(self.parameter.FOLDER, "mask_enhanced.png"))
        config = self.parameter
        for scene_folders in TQDM(
            self.listdir_space_scene(config.FOLDER), desc="Group images"
        ):
            for image in os.listdir(scene_folders):
                if not any(x in image for x in config.IMAGE_FILES):
                    continue
                if not image.endswith(".png") and not image.endswith(".hdr"):
                    continue

                # Create the destination folder if it doesn't exist
                image_group = [x for x in config.IMAGE_FILES if x in image][0]
                destination_folder = os.path.join(
                    config.FOLDER,
                    config.DEST_FOLDER,
                    image_group,
                )
                os.makedirs(destination_folder, exist_ok=True)

                # Copy the image file to the destination folder
                image_path = os.path.join(scene_folders, image)

                image_name = f'{scene_folders.replace(os.path.dirname(__file__),"").replace("/","__")}_{image}'
                destination_path = os.path.join(destination_folder, image_name)
                image = cv2.imread(
                    image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
                )

                if config.RGB_NIR_FUSION_ENABLED and "channel_0" in image_name:
                    image_nir = cv2.imread(image_path.replace("channel_0", "channel_1"))
                    image = self.rgb_nir_fusion(
                        image, image_nir, mask if config.USE_MASK else None
                    )

                if image.dtype == np.uint16:
                    image = (image / 256).astype(np.uint8)
                cv2.imwrite(destination_path, image)


if __name__ == "__main__":
    args = parse_args()
    config = Preprocessing.HyperParamter()
    config.FOLDER = args.folder
    config.DEST_FOLDER = args.dest_folder
    config.RGB_NIR_FUSION_ENABLED = args.rgb_nir_fusion
    config.USE_MASK = args.use_mask
    config.IMAGE_FILES = args.image_files.split(",")
    config.hdr_fusion = args.hdr_fusion
    preprocessing = Preprocessing(config)

    if config.hdr_fusion:
        preprocessing.hdr_fusion_space()
    else:
        preprocessing.group_images()
