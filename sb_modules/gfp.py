import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
import sys
import time
from PIL import Image

from gfpgan import GFPGANer

class GfpClass:
    def __init__(self):
        self.gfp_restorer = None

    def setup(self):
        self.gfp_restorer = GFPGANer(
            model_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "../pretrained_weights/gfp/GFPGANv1.4.pth"
            ),
            device="cuda",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

    def simple_restore(self, img):
        st = time.time()
        if isinstance(img, Image.Image):
            cv2img = np.array(img)
            cv2img = cv2.cvtColor(cv2img, cv2.COLOR_RGB2BGR)
        else:
            cv2img = img

        cropped_faces, restored_faces, restored_img = self.gfp_restorer.enhance(
            cv2img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5,
        )
        print("gfp cost==", time.time() - st)

        if isinstance(img, Image.Image):
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            restored_img = Image.fromarray(restored_img)

        return restored_img
