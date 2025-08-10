"""
MotionBERT: Unified Pretraining for Human Motion Analysis

https://github.com/Walter0807/MotionBERT
"""

import numpy as np
import copy
import random
import torch
import torch.nn as nn
import logging
from functools import partial
from helpers.DSTformer import DSTformer

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

MOTIONBERT_MAP = [
    "HIPS",
    "RIGHT_HIP",
    "RIGHT_KNEE",
    "RIGHT_ANKLE",
    "LEFT_HIP",
    "LEFT_KNEE",
    "LEFT_ANKLE",
    "SPINE",
    "CHEST",
    "JAW",
    "HEAD",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
]


class MotionBert:
    """
    This class imports and executes the model called MotionBert. The model was downloaded from the
    source. It works with a receptive field of 243 frames.
    """

    def __init__(self, probing_point=121):
        # Load MotionBert model with MotionBert arguments for network inference
        self.model = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=256,
            dim_rep=512,
            depth=5,
            num_heads=8,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=243,
            num_joints=17,
        )
        self.probing_point = probing_point
        model_params = 0
        for parameter in self.model.parameters():
            model_params = model_params + parameter.numel()
        logging.info(f"Trainable parameter count: {model_params}")
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        chk_filename = "models/motionbert/FT_MB_lite_MB_ft_h36m_global_lite.bin"
        logging.info(f"Loading checkpoint ´{chk_filename}")
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        state_dict = checkpoint["model_pos"]  # todo adapt for possible gpu use
        state_dict = {
            k.partition("module.")[2]: v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }

        self.model.load_state_dict(state_dict, strict=True)

        logging.info("Testing")
        self.model.eval()

    def normalize_screen_coordinates(self, X, w, h):
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio.
        assert X.shape[-1] == 2
        return X / w * 2 - [1, h / w]

    def get_3d_keypoints(self, input_2D_no, img_size):
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = self.normalize_screen_coordinates(
            input_2D_no, w=img_size[0], h=img_size[1]
        )
        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[
            :, joints_right + joints_left
        ]
        input_2D = np.concatenate(
            (np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0
        )

        input_2D = input_2D[np.newaxis, :, :, :, :]

        test_confidence = np.ones(input_2D.astype("float32").shape)[
            :, :, :, :, 0:1
        ]  # MOTIONBERT CUSTOM ( we need confidence from 2d inference )
        input_2D = np.concatenate(
            (input_2D.astype("float32"), test_confidence), axis=4
        )  # [BS, N, 17, 3]

        input_2D = torch.from_numpy(input_2D.astype("float32"))

        if torch.cuda.is_available():
            input_2D = input_2D.cuda()

        ## estimation
        output_3D_non_flip = self.model(input_2D[:, 0])
        output_3D_flip = self.model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[
            :, :, joints_right + joints_left, :
        ]

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D = output_3D[0, :, :, :]
        post_out = output_3D.cpu().detach().numpy()
        return post_out
