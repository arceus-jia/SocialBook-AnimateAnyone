import os, sys

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, "../"))

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
import glob
import torch.nn.functional as F
from dwpose import DWposeDetector, draw_pose_simple
import cv2
import math

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid
from moviepy.editor import VideoFileClip
import traceback

detector = DWposeDetector()
detector = detector.to(f"cuda")


# 0 鼻 1 脖根 2 左肩 3 左肘 4 左腕 5 右肩 6 右肘 7 右腕
# 8 左胯 9 左膝 10左踝 11 右胯 12 右膝 13右踝
# 14 左眼 15 右眼 16 左耳 17右耳
class TreeNode:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.new_x = point[0]
        self.new_y = point[1]
        self.children = []
        self.parent = None
        self.scale = 1

    def add_child(self, child):
        self.children.append(child)
        child.parent = self


def get_dis(node):
    # todo 肢体缺失
    if not node.parent:
        return
    dis = ((node.x - node.parent.x) ** 2 + (node.y - node.parent.y) ** 2) ** 0.5
    return dis


def get_scale(node, ref_node):
    for child1, child2 in zip(node.children, ref_node.children):
        dis1 = get_dis(child1)
        dis2 = get_dis(child2)
        child1.scale = dis2 / dis1
        get_scale(child1, child2)


def adjust_coordinates(node):
    # node.new_x += offset[0]
    # node.new_y += offset[1]

    if node.parent:
        # 和父亲距离
        dx = node.x - node.parent.x
        dy = node.y - node.parent.y
        # scale
        dx *= node.scale
        dy *= node.scale
        # 新坐标
        new_x = node.parent.new_x + dx
        new_y = node.parent.new_y + dy
        # 仿射
        center = (node.parent.new_x, node.parent.new_y)
        M = cv2.getRotationMatrix2D(center, 0, 1.0)
        new_coordinates = np.dot(np.array([[new_x, new_y, 1]]), M.T)
        # update
        node.new_x, node.new_y = new_coordinates[0, :2]

    for child in node.children:
        adjust_coordinates(child)


def build_tree(pose):

    bodies = pose["bodies"]["candidate"]

    # todo 手，眼睛
    # TODO, 有些节点为空,数值边界
    nodes = [None] * 18
    root = TreeNode(bodies[1])
    nodes[1] = root

    # 脖子到肩膀鼻子腰
    for i in [0, 2, 5, 8, 11]:
        nodes[i] = TreeNode(bodies[i])
        root.add_child(nodes[i])

    # 脸
    for i in [14, 15, 16, 17]:
        nodes[i] = TreeNode(bodies[i])
        nodes[0].add_child(nodes[i])

    # 左臂
    nodes[3] = TreeNode(bodies[3])
    nodes[2].add_child(nodes[3])

    nodes[4] = TreeNode(bodies[4])
    nodes[3].add_child(nodes[4])

    # 右臂
    nodes[6] = TreeNode(bodies[6])
    nodes[5].add_child(nodes[6])

    nodes[7] = TreeNode(bodies[7])
    nodes[6].add_child(nodes[7])

    # 左腿
    nodes[9] = TreeNode(bodies[9])
    nodes[8].add_child(nodes[9])

    nodes[10] = TreeNode(bodies[10])
    nodes[9].add_child(nodes[10])

    # 右腿
    nodes[12] = TreeNode(bodies[12])
    nodes[11].add_child(nodes[12])

    nodes[13] = TreeNode(bodies[13])
    nodes[12].add_child(nodes[13])

    # 手 2 21 2, 0右
    # print ('hands==',pose['hands'])
    # input('x')
    hand_nodes = []
    for single_hand in pose["hands"]:
        single_hand_nodes = [None] * 21
        single_hand_nodes[0] = TreeNode(single_hand[0])
        for i in range(5):
            for j in range(4):
                idx = i * 4 + j + 1
                single_hand_nodes[idx] = TreeNode(single_hand[idx])
                if j == 0:
                    # print('idx==',idx)
                    single_hand_nodes[0].add_child(single_hand_nodes[idx])
                else:
                    single_hand_nodes[idx - 1].add_child(single_hand_nodes[idx])
        hand_nodes.append(single_hand_nodes)

    nodes[7].add_child(hand_nodes[0][0])
    nodes[4].add_child(hand_nodes[1][0])
    nodes = nodes + hand_nodes[0] + hand_nodes[1]

    # 脸
    faces = pose["faces"][0]  # 1 68 2
    # print('faces',faces)
    face_nodes = [None] * 68

    # TODO， 鼻子嘴巴这些可以平均
    for i in range(68):
        if i < 36 or i >= 48:
            face_nodes[i] = TreeNode(faces[i])
            nodes[0].add_child(face_nodes[i])

    # 眼睛
    for i in range(6):
        face_nodes[36 + i] = TreeNode(faces[36 + i])
        nodes[14].add_child(face_nodes[36 + i])

        face_nodes[36 + i + 6] = TreeNode(faces[36 + i + 6])
        nodes[15].add_child(face_nodes[36 + i + 6])
    nodes = nodes + face_nodes

    return nodes


# 算algin想要变成ref的缩放比例
def get_scales(ref_pose, align_pose):
    scales = []
    ref_nodes = build_tree(ref_pose)
    align_nodes = build_tree(align_pose)

    get_scale(align_nodes[1], ref_nodes[1])
    for align_node in align_nodes:
        scales.append(align_node.scale)

    print("scales0==", scales)
    # 两只胳膊scale应当一样,不然有几率会越拉越长
    pairs = [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13], [14, 15], [16, 17]]
    for i, j in pairs:
        s = (scales[i] + scales[j]) / 2
        scales[i] = s
        scales[j] = s

    # 手可以根据肢体长度scale ,不然初始状态影响很大
    scales[18:60] = [(scales[8] + scales[7]) / 2] * 42
    
    scales = [1 if math.isnan(i) or math.isinf(i) else i for i in scales]
    print("scales1==", scales)
    return scales


# 在pose的基础上缩放成ref的尺寸
def align_frame(pose, ref_pose, scales, offset):
    nodes = build_tree(pose)
    for node, scale in zip(nodes, scales):
        node.scale = scale
        # print('scale==',scale)

    adjust_coordinates(nodes[1])
    new_pose = []
    for node in nodes:
        new_pose.append([node.new_x + offset[0], node.new_y + offset[1]])
    return new_pose


def draw_new_pose(pose, subset, H, W):
    bodies = pose[:18]
    hands = [pose[18:39], pose[39:60]]
    faces = [pose[60:128]]

    data = {
        "bodies": {"candidate": bodies, "subset": subset},
        "hands": hands,
        "faces": faces,
    }
    # print('data==',data)
    result = draw_pose_simple(data, H, W)
    return result


def align_image_pose(input_img,ref_img, align_img, W, H,no_face):
    # 统一尺寸(客户端裁剪)
    align_img = crop_center_and_resize(align_img, W, H)
    ref_img = crop_center_and_resize(ref_img, W, H)

    # 获取scale
    ref_pose, _ = get_pose(ref_img)
    # _.save("refpose.jpg")
    align_pose, _ = get_pose(align_img)
    # _.save("alignpose.jpg")
    scales = get_scales(ref_pose, align_pose)

    align_nodes = build_tree(align_pose)
    ref_nodes = build_tree(ref_pose)
    offset = [ref_nodes[1].x - align_nodes[1].x, ref_nodes[1].y - align_nodes[1].y]    

    pose, _ = get_pose(input_img)
    subset = pose["bodies"]["subset"]
    new_pose = align_frame(pose, ref_pose, scales,offset)

    result = draw_new_pose(new_pose, subset, H, W)
    return result

def handle_image(input_img, output_img, ref_img, align_img, W, H):
    result = align_image_pose(input_img,ref_img, align_img, W, H)
    result.save(output_img)


def handle_video(input_video, output_video, ref_img, align_img, W, H,noface):
    # 统一尺寸(客户端裁剪)
    align_img = crop_center_and_resize(align_img, W, H)
    ref_img = crop_center_and_resize(ref_img, W, H)

    # 获取scale
    ref_pose, _ = get_pose(ref_img)
    # _.save("refpose.jpg")
    align_pose, _ = get_pose(align_img)
    # _.save("alignpose.jpg")
    scales = get_scales(ref_pose, align_pose)

    align_nodes = build_tree(align_pose)
    ref_nodes = build_tree(ref_pose)
    offset = [ref_nodes[1].x - align_nodes[1].x, ref_nodes[1].y - align_nodes[1].y]

    video = VideoFileClip(input_video)
    fps = round(video.fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = cv2.VideoCapture(input_video)
    out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    idx = 0

    try:
        while True:
            idx += 1
            success, img = cap.read()
            if not success:
                break
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = crop_center_and_resize(img, W, H)

            pose, _ = get_pose(img)
            subset = pose["bodies"]["subset"]
            new_pose = align_frame(pose, ref_pose, scales, offset)

            result = draw_new_pose(new_pose, subset, H, W)
            # result.save('tmp.jpg')
            # input('x')
            result = np.array(result)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            # print('width,height',width,height,img.shape)
            out.write(result)
    except Exception as e:
        traceback.print_exc()
        print("video error:: 行号--", e.__traceback__.tb_lineno)
    finally:
        cap.release()
        out.release()


# 根据ref 和 align 算出scale, 然后对input的每一帧利用scale计算坐标,ref=xx, align=chen
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-W", type=int, default=512, help="Width")
    parser.add_argument("-H", type=int, default=896, help="Height")
    parser.add_argument("--input", "-i", type=str, help="input video")
    parser.add_argument("--align", "-a", type=str, help="align img")
    parser.add_argument("--ref", "-r", type=str, help="ref img")
    parser.add_argument("--output", "-o", type=str, help="output img or video")

    args = parser.parse_args()
    return args


def get_pose(image):
    result, pose_data = detector(image, only_eye=False)

    candidate = pose_data["bodies"]["candidate"]
    subset = pose_data["bodies"]["subset"]
    # result.save('tmp.jpg')
    # input('x')
    return pose_data, result


def crop_center_and_resize(img, target_width, target_height):

    # 获取原始图像的尺寸
    orig_width, orig_height = img.size

    # 计算裁剪的目标尺寸
    # 首先计算缩放比例
    scale = min(orig_width / target_width, orig_height / target_height)

    # 然后计算裁剪尺寸
    new_width = target_width * scale
    new_height = target_height * scale

    # 计算裁剪框的左上角和右下角坐标
    left = (orig_width - new_width) / 2
    top = (orig_height - new_height) / 2
    right = (orig_width + new_width) / 2
    bottom = (orig_height + new_height) / 2

    # 裁剪图像
    img_cropped = img.crop((left, top, right, bottom))

    # 缩放图像
    img_resized = img_cropped.resize((target_width, target_height), Image.ANTIALIAS)

    return img_resized


def is_image_file(file_path):
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]
    return any(file_path.lower().endswith(ext) for ext in image_extensions)


if __name__ == "__main__":
    args = parse_args()

    ref_img = Image.open(args.ref)
    align_img = Image.open(args.align)

    if is_image_file(args.input):
        input_img = Image.open(args.input)
        handle_image(input_img, args.output, ref_img, align_img, args.W, args.H)
    else:
        handle_video(args.input, args.output, ref_img, align_img, args.W, args.H)


#  python align_pose_full.py --align ../data/align_images/ali1.jpg --ref ../data/images/head2.png --input ../data/videos/ali1.mp4 --output ../data/pose_full/head2_ali1.mp4

# python align_pose_full.py --align ../data/align_images/ali1.jpg --ref ../data/images/bear.jpg --input ../data/frames/ali1/0017.jpg --output tmp.jpg