import os, sys
import io
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname, "./Moore-AnimateAnyone"))
sys.path.append(os.path.join(dirname, "./"))
from datetime import datetime
from pathlib import Path
from typing import List
import uuid
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
import gradio as gr
# from align_pose import handle_video
# from align_pose_full import handle_video
INF_WIDTH = 768
INF_HEIGHT = 768


class Args:
    def __init__(self):
        self.W = 512
        self.H = 896
        self.L = 124
        self.S = 24
        self.O = 4
        self.cfg = 3.5
        self.seed = 42
        self.steps = 20
        self.fps = None
        self.skip = 1
        self.grid = False

        self.pretrained_base_model_path = "./pretrained_weights/stable-diffusion-v1-5/"
        self.pretrained_vae_path = "./pretrained_weights/sd-vae-ft-mse"
        self.image_encoder_path = "./pretrained_weights/image_encoder"
        self.denoising_unet_path = "./pretrained_weights/public_full/denoising_unet.pth"
        self.reference_unet_path = "./pretrained_weights/public_full/reference_unet.pth"
        self.pose_guider_path = "./pretrained_weights/public_full/pose_guider.pth"
        self.motion_module_path = "./pretrained_weights/public_full/motion_module.pth"
        self.is_full_pose = True
        self.inference_config = "./Moore-AnimateAnyone/configs/inference/inference_v2.yaml"
        self.weight_dtype = 'fp16'

    def print_args(self):
        print("Width:", self.W)
        print("Height:", self.H)
        print("Length:", self.L)
        print("Slice:", self.S)
        print("Overlap:", self.O)
        print("Classifier free guidance:", self.cfg)
        print("DDIM sampling steps :", self.steps)
        print("Pretrained base model path:", self.pretrained_base_model_path)
        print("Pretrained VAE path:", self.pretrained_vae_path)
        print("Image encoder path:", self.image_encoder_path)
        print("Denoising U-Net path:", self.denoising_unet_path)
        print("Reference U-Net path:", self.reference_unet_path)
        print("Pose guider path:", self.pose_guider_path)
        print("Motion module path:", self.motion_module_path)
        print("Is full pose:", self.is_full_pose)
        print("Inference config path:", self.inference_config)
        print("Weight data type:", self.weight_dtype)


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


def inference(align_image, input_video, ref_image):
    args = Args()
    print("load===")
    if args.is_full_pose:
        from tools.align_pose_full import handle_video
        pose_folder = 'pose_full'
    else:
        from tools.align_pose import handle_video
        pose_folder = 'pose'
    pose_folder = os.path.join(dirname,'./output/',pose_folder)
    os.makedirs(pose_folder,exist_ok=True)

    if args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = args.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        args.pretrained_base_model_path,
        args.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    pose_guider = PoseGuider(320, block_out_channels=(16, 32, 96, 256)).to(
        dtype=weight_dtype, device="cuda"
    )

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        args.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(args.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(args.reference_unet_path, map_location="cpu"),
    )
    pose_guider.load_state_dict(
        torch.load(args.pose_guider_path, map_location="cpu"),
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

    def convert_to_pil_image(obj):
        print(f"Input object type: {type(obj)}")
        if isinstance(obj, np.ndarray):
            print("Converting NumPy array to PIL Image")
            stream = io.BytesIO()
            Image.fromarray(obj).save(stream, format='PNG')
            stream.seek(0)
            pil_image = Image.open(stream)
            print(f"Output object type: {type(pil_image)}")
            return pil_image
        elif hasattr(obj, 'get_image_data'):
            print("Converting Gradio Image component to PIL Image")
            np_array = obj.get_image_data()
            return convert_to_pil_image(np_array)
        elif isinstance(obj, str):
            print("Read iamge")
            return Image.open(obj).convert("RGB")
        else:
            print("Returning input object as is")
            return obj

    def handle_single(ref_image, input_video, align_image):
        print("handle===", args.motion_module_path)
        align_image_pil = convert_to_pil_image(align_image)
        ref_image_pil = convert_to_pil_image(ref_image)

        ref_image_pil = crop_center_and_resize(
            ref_image_pil, width, height
        )  # 理论上传之前就crop好
        align_image_pil = crop_center_and_resize(align_image_pil, width, height)
        print("----------------")
        # pose

        pose_video_path = os.path.join(pose_folder, f"{str(uuid.uuid4())}.mp4")
        print("pose_video_path==", pose_video_path)
        if not os.path.exists(pose_video_path):
            handle_video(
                input_video,
                pose_video_path,
                ref_image_pil,
                align_image_pil,
                width,
                height,
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
        ).videos

        if args.grid == True:
            video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
            
        video = scale_video(video, width, height)

        m1 = args.pose_guider_path.split(".")[0].split("/")[-1]
        m2 = args.motion_module_path.split(".")[0].split("/")[-1]

        save_dir_name = f"{time_str}-{args.cfg}-{m1}-{m2}"
        save_dir = Path(os.path.join(dirname,"./output/",f"video-{date_str}/{save_dir_name}"))
        save_dir.mkdir(exist_ok=True, parents=True)
        video_path = f"{save_dir}/{str(uuid.uuid4())}_{args.cfg}_{args.seed}_{args.skip}_{m1}_{m2}.mp4"
        save_videos_grid(
            video,
            video_path,
            n_rows=3,
            fps=src_fps if args.fps is None else args.fps,
        )
        return gr.Video.update(value=video_path)

    return handle_single(ref_image, input_video, align_image)
    

def clear_media(align_image, input_video, ref_image, output_video):
    return gr.Image.update(value=None), gr.Video.update(value=None), gr.Image.update(value=None), gr.Video.update(value=None)
with gr.Blocks() as demo:
    gr.Markdown(
        """
# SocialBook-AnimateAnyone
## We are SocialBook, you can experience our product through the link.
<div style="display: flex; align-items: center;">
  <a href="https://socialbook.io/" style="margin-right: 20px;">
    <img src="https://d35b8pv2lrtup8.cloudfront.net/assets/img/socialbook_logo.2020.357eed90add7705e54a8.svg" alt="SocialBook" width="200" height="100">
  </a>
  <a href="https://dreampal.socialbook.io/">
    <img src="https://d35b8pv2lrtup8.cloudfront.net/assets/img/logo.ce05d254bbdb2d417c4f.svg" alt="DreamPal" width="200" height="100">
  </a>
</div>

The first complete animate anyone code repository

[Shunran Jia](https://github.com/arceus-jia), Zhengyan Tong (Shanghai Jiao Tong University), [Xuanhong Chen](https://github.com/neuralchen), [Chen Wang](https://socialbook.io/), [Chenxi Yan](https://github.com/todochenxi)
        """)
    with gr.Row():
        with gr.Column():
            align_image = gr.Image(souce=["upload", "clipboard"], label="Align Image(a standard frame of a person's pose from the dance_video)")
            input_video = gr.Video(souce=["upload", "clipboard"],label="Dance Video")  
        with gr.Column():  
            ref_image = gr.Image(souce=["upload", "clipboard"],label="New Image")
            output_video = gr.Video(label="Result")
    with gr.Row():
        clean = gr.Button("Clean")
        run = gr.Button("Generate")
    ex_data = [
        ["data/align_images/ali11.jpg", "data/videos/ali11.mp4", "data/images/head2.png"],
        ["data/align_images/ali11.jpg", "data/videos/ali12.mp4", "data/images/asuna3.jpg"],
        ["data/align_images/ali11.jpg", "data/videos/ali11.mp4", "data/images/mbg1.jpg"]
    ]
    examples_component = gr.Examples(examples=ex_data, inputs=[align_image, input_video, ref_image], outputs=[output_video], fn=inference, label="Examples", cache_examples=False, run_on_click=True)
    clean.click(clear_media, [align_image, input_video, ref_image, output_video], [align_image, input_video, ref_image, output_video])
    run.click(inference, [align_image, input_video, ref_image], [output_video])


demo.launch(share=False,
             debug=False,
             server_name="0.0.0.0",
             server_port=7860
)
    


