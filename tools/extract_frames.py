import os
import sys
import cv2
import subprocess
import argparse
import time
from moviepy.editor import VideoFileClip
import traceback
import glob
from concurrent.futures import ThreadPoolExecutor
import concurrent

parser = argparse.ArgumentParser(description="animate demo")
parser.add_argument("--input", default=None, help="input path")
parser.add_argument("--output", default=None, help="output path")
parser.add_argument("--max-cnt", default=20, help="output path")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

def handle_video(input_video):
    st = time.time()
    print('handle===', input_video)

    cap = cv2.VideoCapture(input_video)

    idx = 0

    out = None
    try:
        while True:
            idx += 1
            success, img = cap.read()
            if not success:
                break
            if img is None:
                continue

            cv2.imwrite(os.path.join(args.output,'%04d.jpg' % idx),img)
            if idx >= int(args.max_cnt):
                break

    except Exception as e:
        print("video error:: 行号--", e.__traceback__.tb_lineno)
        traceback.print_exc()
    finally:
        cap.release()

    print('cost::', time.time() - st)

if __name__ == "__main__":
    handle_video(args.input)

#  python extract_frames.py --input ../data/videos/ali1.mp4 --output ../data/frames/ali1