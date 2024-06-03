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
import onnxruntime

import insightface
from insightface.app import FaceAnalysis


class InswapperClass:
    def __init__(self):
        self.fa = None
        self.face_swapper = None

        self.dirname = os.path.dirname(os.path.abspath(__file__))
        self.base_model_path = os.path.join(self.dirname, "../pretrained_weights/inswapper")

    def setup(self):

        providers = onnxruntime.get_available_providers()
        print('providers==',providers)
        # ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
        # providers = ['CPUExecutionProvider']
        det_size = (320, 320)
        self.fa = FaceAnalysis(
            name="buffalo_l", root=self.base_model_path, providers=providers
        )
        self.fa.prepare(ctx_id=0, det_size=det_size)

        self.face_swapper = insightface.model_zoo.get_model(
            os.path.join(self.base_model_path, "inswapper_128.onnx")
        )
        print("providers==", providers)

    def get_one_face(self, frame: np.ndarray):
        face = self.fa.get(frame)
        try:
            return min(face, key=lambda x: x.bbox[0])
        except ValueError:
            return None

    def get_many_faces(self, frame: np.ndarray,max_cnt=3):
        try:
            face = self.fa.get(frame)
            face = sorted(face, key=lambda x: -(x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))[:max_cnt]
            return sorted(face, key=lambda x: -x.bbox[0])
        except IndexError:
            return None

    def swap_face(self,
                source_face,
                target_faces,
                target_index,
                temp_frame):
        """
        paste source_face on target image
        """
        target_face = target_faces[target_index]

        return self.face_swapper.get(temp_frame, target_face, source_face, paste_back=True)

    # 把target_img的人脸换成resource_imgs的
    def process(self, resource_imgs, target_img):
        st = time.time()
        # target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        target_faces =  self.get_many_faces(target_img)
        source_faces = []
        for img in resource_imgs:
            # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            source_faces.append(self.get_one_face(img))


        tmp_img = target_img.copy()
        idx = 0
        for source_face in source_faces:
            tmp_img = self.swap_face(source_face, target_faces,idx,tmp_img)
            idx += 1

        # result_img = Image.fromarray(cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB))
        result_img = tmp_img

        print('swap cost::', time.time()-st)
        return result_img
