# SocialBook-AnimateAnyone
We are SocialBook, you can experience our other products through these links.
<div style="display: flex; align-items: center;">
  <a href="https://socialbook.io/" style="margin-right: 20px;">
    <img src="https://d35b8pv2lrtup8.cloudfront.net/assets/img/socialbook_logo.2020.357eed90add7705e54a8.svg" alt="SocialBook" width="200" height="100">
  </a>
  <a href="https://dreampal.socialbook.io/">
    <img src="https://d35b8pv2lrtup8.cloudfront.net/assets/img/logo.ce05d254bbdb2d417c4f.svg" alt="DreamPal" width="200" height="100">
  </a>
</div>
The first complete animate anyone code repository

Shunran Jia,[Xuanhong Chen](https://github.com/neuralchen),
Chen Wang,
[Chenxi Yan](https://github.com/todochenxi)


**_We plan to provide a complete set of animate anyone training code and high-quality training data in the next few days to help the community implement its own high-performance animate anyone training._**

## Overview
[SocialBook-AnimateAnyone](https://github.com/arceus-jia/SocialBook-AnimateAnyone) is a generative model for converting images into videos, specifically designed to create virtual human videos driven by poses. 

We have implemented this model based on the [AnimateAnyone](https://github.com/HumanAIGC/AnimateAnyone)  paper and further developed it based on  [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone).We are very grateful for their contributions.

### Our contributions include:
- Conducting secondary development on Moore-AnimateAnyone, where we applied various tricks and different training parameters and approaches compared to Moore, resulting in more stable generation outcomes.
- Performing pose alignment work, allowing for better consistency across different facial expressions and characters during inference.
- We plan to open-source our model along with detailed training procedures.


## Demos
<table class="center">
<tr>
    <td width=50% style="border: none">
    <video controls autoplay loop src="https://github.com/arceus-jia/SocialBook-AnimateAnyone/assets/5162767/8754fd0a-10b2-441f-aacb-89ac52ceb4c1" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/arceus-jia/SocialBook-AnimateAnyone/assets/5162767/bb3060a8-3b38-42c4-812d-65694bb3c0b6" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=50% style="border: none">
    <video controls autoplay loop src="https://github.com/arceus-jia/SocialBook-AnimateAnyone/assets/5162767/187b5ce0-b064-417f-a59b-80f48719de97" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/arceus-jia/SocialBook-AnimateAnyone/assets/5162767/1066bc5f-a8e9-441f-b709-7103c74620c5" muted="false"></video>
    </td>
</tr>
</table>

<!-- ## Try it online
You can try it out on our demos page now!
![1277035298](https://github.com/arceus-jia/SocialBook-AnimateAnyone/assets/5162767/c0c7cc10-27e5-4abd-9afa-8e711a1d1a51)

<a href = 'http://animateanyone.socialbook.io:48001'>Click to try</a> -->

## Windows整合包
感谢b站用户 PeterPan369 为此项目制作的整合包。有需要的朋友可以自行下载使用，建议使用7-zip解压
https://pan.baidu.com/s/1Q_aDp_N2CSz-rqk7gIfKiQ?pwd=3u82 


## TODO
- [x] Release Inference Dode
- [x] Gradio Demo
- [x] Add Face Enhancement
- [ ] Build online test page
- [ ] ReleaseTraining Code And Data
## News
- [05/27/2024] Release Inference Code
- [05/31/2024] Add a Gradio Demo
- [06/03/2024] Add facial repair
- [06/05/2024] Release a demo page
# Getting Started

## Installation

### Clone repo
```bash
git clone git@github.com:arceus-jia/SocialBook-AnimateAnyone.git --recursive
```

### Setup environment
```bash
conda create -n aa python=3.10
conda activate aa
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

### Download weights
```bash
python tools/download_weights.py

#optional
mkdir -p pretrained_weights/inswapper
wget -O pretrained_weights/inswapper/inswapper_128.onnx  https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

mkdir -p pretrained_weights/gfp
wget -O pretrained_weights/gfp/GFPGANv1.4.pth https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

```

`pretrained_weights` structure is:
```
./pretrained_weights/
|-- public_full
|   |-- denoising_unet.pth
|   |-- motion_module.pth
|   |-- pose_guider.pth
|   └── reference_unet.pth
|-- stable-diffusion-v1-5
|   └── unet
|       |-- config.json
|       └── diffusion_pytorch_model.bin
|-- image_encoder
|   |-- config.json
|   └── pytorch_model.bin
└── sd-vae-ft-mse
    |-- config.json
    └── diffusion_pytorch_model.bin
```

```
中国的同学们也可以使用百度网盘直接下载所有权重文件
链接: https://pan.baidu.com/s/1gyWmFiEaOMw-vnuRr6UJew  密码: d669

```


## Quickstart
### Inference
#### Prepare Data
Place the image, dance_video, and aligned_dance_image you prepared into the 'images', 'videos', and 'align_images' folders under the 'data' directory. (In general, 'dance_align_image' refers to a standard frame of a person's pose from the 'dance_video'.) 
```
./data/
|-- images
|   └── human.jpg
└── videos
    └── dance.mp4
└── align_images
    └── dance.jpg

```
And modify the 'script/test_video.yaml' file according to your configuration.


#### Run inference
```bash
cd script
python test_video.py -L 48 --grid
```
Parameters:
```
-L: Frames count
--grid: Enable grid overlay with pose/original_image
--seed: seed
-W: video width
-H: video height
--skip: frame interpolation
```
And you can see the output results in ```./output/```

If you want to do facial repair on a video (only for videos of REAL PERSON)
```bash
python restore_face.py --ref_image xxx.jpg --input xxx.mp4 --output xxx.mp4
```

#### Gradio （beta, under developement）
```bash
python app.py
```


### Training
