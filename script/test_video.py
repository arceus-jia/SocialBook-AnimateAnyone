import os, sys

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, "../Moore-AnimateAnyone"))
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
from dwpose import DWposeDetector
import cv2
import math

from configs.prompts.test_cases import TestCasesDict
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_long import Pose2VideoPipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

# from align_pose import handle_video
# from align_pose_full import handle_video

INF_WIDTH = 768
INF_HEIGHT = 768


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_video.yaml")
    parser.add_argument("-W", type=int, default=512, help="Width")
    parser.add_argument("-H", type=int, default=896, help="Height")
    parser.add_argument("-L", type=int, default=124, help="video frame length")
    parser.add_argument("-S", type=int, default=24, help="video slice frame number")
    parser.add_argument(
        "-O", type=int, default=4, help="video slice overlap frame number"
    )

    parser.add_argument(
        "--cfg", type=float, default=3.5, help="Classifier free guidance"
    )
    parser.add_argument("--seed", type=int, default=42, help="DDIM sampling steps")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=int)

    parser.add_argument("--skip", type=int, default=1)  # 插帧
    parser.add_argument(
        "--grid",
        default=False,
        action="store_true",
        help="grid",
    )     
    args = parser.parse_args()

    print("Width:", args.W)
    print("Height:", args.H)
    print("Length:", args.L)
    print("Slice:", args.S)
    print("Overlap:", args.O)
    print("Classifier free guidance:", args.cfg)
    print("DDIM sampling steps :", args.steps)

    return args


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


def scale_video(video, width, height):
    # 重塑video张量以合并batch和frames维度
    video_reshaped = video.view(
        -1, *video.shape[2:]
    )  # [batch*frames, channels, height, width]

    # 使用双线性插值缩放张量
    # 注意：'align_corners=False'是大多数情况下的推荐设置，但你可以根据需要调整它
    scaled_video = F.interpolate(
        video_reshaped, size=(height, width), mode="bilinear", align_corners=False
    )

    # 将缩放后的张量重塑回原始维度
    scaled_video = scaled_video.view(
        *video.shape[:2], scaled_video.shape[1], height, width
    )  # [batch, frames, channels, height, width]

    return scaled_video


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    print("load===")
    pose_type = config.pose_type

    if pose_type == "full":
        from tools.align_pose_full import handle_video

        pose_folder = "pose_full"
    else:
        from tools.align_pose import handle_video

        if pose_type == "noface":
            pose_folder = "pose_noface"
        else:
            pose_folder = "pose"

    pose_folder = os.path.join(dirname,'../output/',pose_folder)
    os.makedirs(pose_folder,exist_ok=True)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(config.pose_guider_path, map_location="cpu"),
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")

    def handle_single(ref_image_path, input_video_path, align_image_path):
        print("handle===", ref_image_path, input_video_path, config.motion_module_path)

        ref_name = Path(ref_image_path).stem
        pose_name = Path(input_video_path).stem.replace("_kps", "")

        align_image_pil = Image.open(align_image_path).convert("RGB")
        ref_image_pil = Image.open(ref_image_path).convert("RGB")

        ref_image_pil = crop_center_and_resize(
            ref_image_pil, width, height
        )  # 理论上传之前就crop好
        align_image_pil = crop_center_and_resize(align_image_pil, width, height)

        # pose

        pose_video_path = os.path.join(pose_folder, f"{ref_name}_{pose_name}.mp4")
        print("pose_video_path==", pose_video_path)
        if not os.path.exists(pose_video_path):
            handle_video(
                input_video_path,
                pose_video_path,
                ref_image_pil,
                align_image_pil,
                width,
                height,
                pose_type == 'noface'
            )

        pose_list = []
        pose_tensor_list = []
        pose_images = read_frames(pose_video_path)
        src_fps = get_fps(pose_video_path)
        print(f"pose video has {len(pose_images)} frames, with {src_fps} fps")
        L = min(args.L, len(pose_images))
        pose_transform = transforms.Compose(
            [transforms.Resize((INF_HEIGHT, INF_WIDTH)), transforms.ToTensor()]
        )

        pose_images = pose_images[:: args.skip + 1]
        src_fps = src_fps // (args.skip + 1)
        L = L // ((args.skip + 1))

        for pose_image_pil in pose_images[:L]:
            # 理论上wh和pose一致，最多缩放一下
            pose_image_pil = crop_center_and_resize(pose_image_pil, width, height)

            pose_tensor_list.append(pose_transform(pose_image_pil))
            pose_list.append(pose_image_pil)
            pose_image_pil = pose_image_pil.resize((INF_WIDTH, INF_HEIGHT))

        ref_image_pil = ref_image_pil.resize((INF_WIDTH, INF_HEIGHT))

        ref_image_tensor = pose_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)  # (1, c, 1, h, w)
        ref_image_tensor = repeat(
            ref_image_tensor, "b c f h w -> b c (repeat f) h w", repeat=L
        )

        pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
        pose_tensor = pose_tensor.transpose(0, 1)
        pose_tensor = pose_tensor.unsqueeze(0)

        video = pipe(
            ref_image_pil,
            pose_list,
            INF_WIDTH,
            INF_HEIGHT,
            L,
            args.steps,
            args.cfg,
            generator=generator,
            context_frames=args.S,
            context_stride=1,
            context_overlap=args.O,
            use_clip=config.use_clip
        ).videos

        if args.grid == True:
            video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
            
        video = scale_video(video, width, height)

        m1 = config.pose_guider_path.split(".")[0].split("/")[-1]
        m2 = config.motion_module_path.split(".")[0].split("/")[-1]

        save_dir_name = f"{time_str}-{args.cfg}-{m1}-{m2}"
        save_dir = Path(os.path.join(dirname,"../output/",f"video-{date_str}/{save_dir_name}"))
        save_dir.mkdir(exist_ok=True, parents=True)

        save_videos_grid(
            video,
            f"{save_dir}/{ref_name}_{pose_name}_{args.cfg}_{args.seed}_{args.skip}_{m1}_{m2}.mp4",
            n_rows=3,
            fps=src_fps if args.fps is None else args.fps,
        )

    for ref_image_path_dir in config["test_cases"].keys():
        if os.path.isdir(ref_image_path_dir):
            ref_image_paths = glob.glob(os.path.join(ref_image_path_dir, "*.jpg"))
        else:
            ref_image_paths = [ref_image_path_dir]
        for ref_image_path in ref_image_paths:
            poses_path = config["test_cases"][ref_image_path_dir]
            pose_video_path = poses_path[0]
            align_image_path = poses_path[1]
            handle_single(ref_image_path, pose_video_path, align_image_path)


if __name__ == "__main__":
    main()

# python test_video.py --config test_video.yaml -W 512 -H 784 -L 48
