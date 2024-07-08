import torch
from modules.depth.raft_stereo.core.raft_stereo import RAFTStereo, autocast
import numpy as np
from modules.depth.raft_stereo.core.utils.utils import InputPadder


class RaftStereoWrapper:

    def __init__(self):

        args = type("", (), {})()

        args.hidden_dims = [128, 128, 128]
        args.corr_levels = 4
        args.corr_radius = 4
        args.n_downsample = 3
        args.context_norm = "instance"
        args.n_gru_layers = 2
        args.shared_backbone = True
        args.mixed_precision = True
        args.corr_implementation = "reg"
        args.slow_fast_gru = True
        self.model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        args.restore_ckpt = "modules/depth/raft_stereo/models/raftstereo-realtime.pth"
        checkpoint = torch.load(args.restore_ckpt)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.cuda()
        self.model.eval()

    def __preprocess_image(self, image: np.ndarray):
        print(image.shape, image.min(), image.max(), image.mean())
        image = image / image.max() * 65535
        img = torch.from_numpy(image).permute(2, 0, 1).float()
        return img[None].cuda()
        image = np.moveaxis(image, -1, 0)
        image = image[np.newaxis, ...]
        image = image / image.max()
        image = torch.tensor(image).cuda()
        return image

    def __call__(self, left: np.ndarray, right: np.ndarray):
        left = self.__preprocess_image(left)
        right = self.__preprocess_image(right)
        padder = InputPadder(left.shape, divis_by=32)
        left, right = padder.pad(left, right)
        # print("Tensor shape", left.shape, left.mean(), left.min(), left.max())
        with autocast(enabled=True):
            _, flow_pr = self.model(left, right, test_mode=True)

        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0).detach().numpy()
        flow_pr = np.moveaxis(flow_pr, 0, -1)
        return -flow_pr
