# clone
```bash
git clone git@github.com:arceus-jia/SocialBook-AnimateAnyone.git --recusive
```

# setup env
```bash
conda create -n aa python=3.10
pip install -r requirements.txt
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"
```

# inference
```bash
cd script
python test_video.py
```