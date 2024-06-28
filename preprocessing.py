import os
from tqdm import tqdm
import cv2
import numpy as np
from fusion import fusion
import argparse


_IMAGE_FILES = [
    "jai_1600_left_channel_0.png",
    "jai_1600_left_channel_1.png",
    "jai_1600_right_channel_0.png",
    "jai_1600_right_channel_1.png",
    # "oakd_left",
    # "oakd_right",
    # "oakd_rgb",
]


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
        default="/images_origin/",
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


class Preprocessing:
    class HyperParamter:
        FOLDER = "lab0624"
        DEST_FOLDER = "images_origin"
        RGB_NIR_FUSION_ENABLED = False
        USE_MASK = False
        IMAGE_FILES = _IMAGE_FILES

    def __init__(self, hyperparameter: HyperParamter):
        self.parameter = hyperparameter

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

    def hdr_fusion(self, image_file_path: str):
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
        exposure_times.sort()
        image_path_list = [
            image_file_name + "_" + str(x) + ".png" for x in exposure_times
        ]
        exposure_times = np.array(
            exposure_times,
            dtype=np.float32,
        )

        images = [
            cv2.imread(
                os.path.join(image_folder, x), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
            )
            for x in image_path_list
        ]

        print("Exposure times:", exposure_times, exposure_times.dtype)

        images = [
            cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for x in images
        ]
        calibrate = cv2.createCalibrateDebevec()
        response = calibrate.process(images, exposure_times)

        merge = cv2.createMergeDebevec()
        hdr = merge.process(images, exposure_times, response)

        if len(hdr.shape) == 2:
            hdr = cv2.cvtColor(hdr, cv2.COLOR_GRAY2BGR)

        tonemap = cv2.createTonemap(2.2)
        ldr = tonemap.process(hdr)

        mertge_mertges = cv2.createMergeMertens()
        fusion = mertge_mertges.process(images)
        cv2.imwrite(image_file_path.replace(".png", "_sdr.png"), ldr * 255)
        cv2.imwrite(image_file_path.replace(".png", "_hdr.hdr"), hdr)
        cv2.imwrite(image_file_path.replace(".png", "_fusion.png"), fusion * 255)

    def group_images(
        self,
    ):
        config = self.parameter
        mask = cv2.imread(
            os.path.join(config.FOLDER, "mask_enhanced.png"), cv2.IMREAD_GRAYSCALE
        )
        for capture_folders in tqdm(os.listdir(config.FOLDER)):
            if not capture_folders.isdigit():
                continue
            if not os.path.isdir(os.path.join(config.FOLDER, capture_folders)):
                continue
            for scene_folders in tqdm(
                os.listdir(os.path.join(config.FOLDER, capture_folders))
            ):
                if not scene_folders.isdigit():
                    continue
                for image in os.listdir(
                    os.path.join(config.FOLDER, capture_folders, scene_folders)
                ):
                    if not any(x in image for x in config.IMAGE_FILES):
                        continue
                    if not image.endswith(".png"):
                        continue

                    if args.hdr_fusion:
                        self.hdr_fusion(
                            f"{config.FOLDER}/{capture_folders}/{scene_folders}/{image}"
                        )
                        continue

                    # Create the destination folder if it doesn't exist

                    destination_folder = os.path.join(
                        config.FOLDER,
                        config.DEST_FOLDER,
                        image.split("_channel_")[0],
                    )
                    os.makedirs(destination_folder, exist_ok=True)

                    # Copy the image file to the destination folder
                    image_path = os.path.join(
                        config.FOLDER, capture_folders, scene_folders, image
                    )

                    image_name = f"{capture_folders}_{scene_folders}_{image}"
                    destination_path = os.path.join(destination_folder, image_name)
                    image = cv2.imread(image_path)

                    if config.RGB_NIR_FUSION_ENABLED and "channel_0" in image_name:
                        image_nir = cv2.imread(
                            image_path.replace("channel_0", "channel_1")
                        )
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
    preprocessing = Preprocessing(config)

    preprocessing.group_images()
