import gradio as gr
import cv2
from PIL import Image as PILImage

def get_image(input_video):
   video = cv2.VideoCapture(input_video)
   ret, first_frame = video.read()
   if ret:
       # 转换OpenCV图像为PIL图像
       pil_image = PILImage.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
       return pil_image

with gr.Blocks() as app:
   video = gr.Video()
   image = gr.Image()
   video.change(fn=get_image, inputs=video, outputs=image)
app.launch(debug=True)