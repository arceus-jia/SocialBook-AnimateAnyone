import os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import argparse
import time
import cv2
from moviepy.editor import VideoFileClip

from sb_modules.gfp import GfpClass
from sb_modules.inswapper import InswapperClass

gfp = GfpClass()
gfp.setup()
inswapper = InswapperClass()
inswapper.setup()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_image", "-r", type=str, help="ref image")
    parser.add_argument("--input", "-i", type=str, help="input video")
    parser.add_argument("--output", "-o", type=str, help="output video")
    args = parser.parse_args()
    return args


def handle_video(ref_image_path, input_video_path, output_video_path):
    st = time.time()
    print("handle===", input_video_path)

    ref_image = cv2.imread(ref_image_path)

    video = VideoFileClip(input_video_path)
    width, height = video.size
    fps = round(video.fps)
    print('video..',fps,width,height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = cv2.VideoCapture(input_video_path)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    idx = 0
    try:
        while True:
            idx += 1
            print('process==',idx)
            success, img = cap.read()
            if not success:
                break
            if img is None:
                continue
            img = inswapper.process([ref_image], img)
            img = gfp.simple_restore(img)
            out.write(img)

    except Exception as e:
        print("video error:: 行号--", e.__traceback__.tb_lineno)
        traceback.print_exc()
    finally:
        cap.release()
        out.release()

    print("cost::", time.time() - st)


if __name__ == "__main__":
    args = parse_args()
    handle_video(args.ref_image, args.input, args.output)
