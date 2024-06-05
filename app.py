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
import argparse
import time

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="script/gradio_config.yaml")
    parser.add_argument("--fps", type=int)
    parser.add_argument(
        "--grid",
        default=False,
        action="store_true",
        help="grid",
    )      
    parser.add_argument("--share", default=False, action="store_true")
    parser.add_argument('--port', type=int, default=7860)

    args = parser.parse_args()
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


def inference(align_image, input_video, ref_image, W, H, cfg, seed, steps, skip):
    print("parms------------>", W, H, cfg, seed, skip)
    W, H, cfg, seed, steps, skip = int(W), int(H), float(cfg), int(seed), int(steps), int(skip) 
    args = parse_args()
    config = OmegaConf.load(args.config)
    print("load===")
    if config.is_full_pose:
        from tools.align_pose_full import handle_video
        pose_folder = 'pose_full'
    else:
        from tools.align_pose import handle_video
        pose_folder = 'pose'
    pose_folder = os.path.join(dirname,'./output/',pose_folder)
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

    generator = torch.manual_seed(seed)

    width, height = W, H

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
        print("handle===", config.motion_module_path)
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
        L = min(config.L, len(pose_images))
        pose_transform = transforms.Compose(
            [transforms.Resize((INF_HEIGHT, INF_WIDTH)), transforms.ToTensor()]
        )

        pose_images = pose_images[:: skip + 1]
        src_fps = src_fps // (skip + 1)
        L = L // ((skip + 1))

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
            steps,
            cfg,
            generator=generator,
            context_frames=24, # video slice frame number
            context_stride=1,
            context_overlap=4, # video slice overlap frame number
        ).videos

        if args.grid == True:
            video = torch.cat([ref_image_tensor, pose_tensor, video], dim=0)
            
        video = scale_video(video, width, height)

        m1 = config.pose_guider_path.split(".")[0].split("/")[-1]
        m2 = config.motion_module_path.split(".")[0].split("/")[-1]

        save_dir_name = f"{time_str}-{cfg}-{m1}-{m2}"
        save_dir = Path(os.path.join(dirname,"./output/",f"video-{date_str}/{save_dir_name}"))
        save_dir.mkdir(exist_ok=True, parents=True)
        video_path = f"{save_dir}/{str(uuid.uuid4())}_{cfg}_{seed}_{skip}_{m1}_{m2}.mp4"
        save_videos_grid(
            video,
            video_path,
            n_rows=3,
            fps=src_fps if args.fps is None else args.fps,
        )
        return gr.Video.update(value=video_path)

    return handle_single(ref_image, input_video, align_image)
    


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    def clear_media(align_image, input_video, ref_image, output_video):
        return gr.Image.update(value=None), gr.Video.update(value=None), gr.Image.update(value=None), gr.Video.update(value=None)

    def get_image(input_video):
        st = time.time()
        video = cv2.VideoCapture(input_video)
        ret, first_frame = video.read()
        if ret:
            # 转换OpenCV图像为PIL图像
            pil_image = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            video.release()
            print("----------------->",time.time() -st)
            return gr.Image.update(value=pil_image)

    with gr.Blocks() as demo:
        gr.Markdown(
            """
    # SocialBook-AnimateAnyone
    ## We are SocialBook, you can experience our other products through these links.
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
                input_video = gr.Video(sources=["upload", "clipboard"],label="Dance Video")
                align_image = gr.Image(sources=["upload", "clipboard"], label="Align Image")  
            with gr.Column(): 
                output_video = gr.Video(label="Result", interactive=False) 
                ref_image = gr.Image(sources=["upload", "clipboard"],label="New Image")
        with gr.Row():
            W = gr.Textbox(label="Width", value=512)
            H = gr.Textbox(label="Height", value=896)
            cfg = gr.Textbox(label="cfg(Classifier free guidance)", value=3.5)
            seed = gr.Textbox(label="seed(DDIM sampling steps)", value=42)
            steps = gr.Textbox(label="steps", value=20)
            skip = gr.Textbox(label="skip(Frame Insertion)", value=1)
        with gr.Row():
            get_align_image = gr.Button("Get Align Image(get the aligned image from dance video)")
            clean = gr.Button("Clean")
            run = gr.Button("Generate")
        ex_data = OmegaConf.to_container(config.examples)
        examples_component = gr.Examples(examples=ex_data, inputs=[align_image, input_video, ref_image], outputs=[output_video], fn=inference, label="Examples", cache_examples=False, run_on_click=True)
        clean.click(clear_media, [align_image, input_video, ref_image, output_video], [align_image, input_video, ref_image, output_video])
        run.click(inference, [align_image, input_video, ref_image, W, H, cfg, seed, steps, skip], [output_video])
        get_align_image.click(get_image, input_video, align_image)
    demo.queue()
    demo.launch(share=args.share,
                debug=True,
                server_name="0.0.0.0",
                server_port=args.port
    )


if __name__ == "__main__":
    main()


    


