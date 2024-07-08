# from module.depth.stereo_transformer_module

import gdown
from .stereo_transformer_module.dataset.preprocess import *
from .stereo_transformer_module.utilities.misc import *
from .stereo_transformer_module.module.sttr import STTR
import torch
import os
import cv2


class StereoTransformer:
    def __init__(self):
        self.model = STTR(self.get_args()).cuda().eval()
        self.download_model()
        self.load_pretrained()

    def download_model(self):
        model_file = "kitti_finetuned_model.pth.tar"
        if os.path.exists(model_file):
            return
        # url = "https://drive.google.com/uc?id=1MW5g1LQ1RaYbqeDS2AlHPZ96wAmkFG_O"
        url = "https://drive.google.com/uc?id=1UUESCCnOsb7TqzwYMkVV3d23k8shxNcE"
        gdown.download(url, "kitti_finetuned_model.pth.tar", quiet=False)

    def get_args(self):
        args = type("", (), {})()  # create empty args
        args.channel_dim = 128
        args.position_encoding = "sine1d_rel"
        args.num_attn_layers = 6
        args.nheads = 4
        args.regression_head = "ot"
        args.context_adjustment_layer = "cal"
        args.cal_num_blocks = 8
        args.cal_feat_dim = 16
        args.cal_expansion_ratio = 4
        return args

    def load_pretrained(self):
        model = STTR(self.get_args())
        model_pretrained = torch.load("kitti_finetuned_model.pth.tar")
        model.load_state_dict(model_pretrained["state_dict"], strict=False)

    def normalize_image(self, image):
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        else:
            pass

        mean = np.mean(image, axis=(1, 2), keepdims=True)
        var = np.var(image, axis=(1, 2), keepdims=True)

        image = (image - mean) / np.sqrt(var + 1e-6) * 3

        return image

    def prepare_input(self, left, right, scale=4):
        left = cv2.resize(left, (left.shape[1] // scale, left.shape[0] // scale))
        right = cv2.resize(right, (right.shape[1] // scale, right.shape[0] // scale))
        h, w, _ = left.shape
        bs = 1
        downsample = 3
        col_offset = int(downsample / 2)
        row_offset = int(downsample / 2)
        sampled_cols = (
            torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
        )
        sampled_rows = (
            torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()
        )
        left = torch.tensor(left.transpose(2, 0, 1)[None,].astype(np.float32)).cuda()
        right = torch.tensor(right.transpose(2, 0, 1)[None,].astype(np.float32)).cuda()
        input_data = NestedTensor(
            left,
            right,
            sampled_cols=sampled_cols,
            sampled_rows=sampled_rows,
        )
        return input_data

    def process(self, left, right):
        left = self.normalize_image(left)
        right = self.normalize_image(right)
        scale = 2
        input_data = self.prepare_input(left, right, scale)
        output = self.model(input_data)
        disp_pred = output["disp_pred"].data.cpu().numpy()[0]
        occ_pred = output["occ_pred"].data.cpu().numpy()[0] > 0.5
        disp_pred[occ_pred] = 0.0
        disp_pred = cv2.resize(disp_pred, (left.shape[1], left.shape[0]))
        return disp_pred


if __name__ == "__main__":
    depth = StereoTransformer()
