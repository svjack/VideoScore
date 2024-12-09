'''
python video_regression_script.py "" xiangling_mp4_dir_tiny

1. **visual quality**:
   - **涵义**: 视觉质量
   - **解释**: 视频在清晰度、分辨率、亮度和色彩等方面的质量。这个维度评估视频的视觉表现，包括图像的清晰度、色彩的准确性和整体的视觉吸引力。

2. **temporal consistency**:
   - **涵义**: 时间一致性
   - **解释**: 视频中物体或人物的一致性。这个维度评估视频在时间上的连贯性，即视频中的物体或人物在不同帧之间是否保持一致，没有明显的跳跃或不连贯的现象。

3. **dynamic degree**:
   - **涵义**: 动态程度
   - **解释**: 视频中动态变化的程度。这个维度评估视频的动态性，即视频中物体或场景的变化程度，包括运动的频率和幅度。

4. **text-to-video alignment**:
   - **涵义**: 文本与视频的对齐
   - **解释**: 文本提示与视频内容之间的对齐程度。这个维度评估视频内容与给定文本提示之间的匹配程度，即视频是否准确地反映了文本提示所描述的内容。

5. **factual consistency**:
   - **涵义**: 事实一致性
   - **解释**: 视频内容与常识和事实知识的一致性。这个维度评估视频内容是否符合常识和事实知识，即视频中的内容是否真实可信，没有明显的逻辑错误或与现实不符的情况。

import pandas as pd
edf = pd.read_csv("evaluation_results.csv")
edf.describe()

print(edf.sort_values(by = "temporal consistency", ascending = True).head(5).to_markdown())

|    | video_name                                                                                         |   visual quality |   temporal consistency |   dynamic degree |   text-to-video alignment |   factual consistency |
|---:|:---------------------------------------------------------------------------------------------------|-----------------:|-----------------------:|-----------------:|--------------------------:|----------------------:|
| 26 | solo,Xiangling,_shave_with_a_razor__genshin_impact__,1girl,highres,_seed_3140464511.mp4            |             2.8  |                   1.14 |             2.97 |                      2.78 |                  1.26 |
|  0 | solo,Xiangling,_carry_money_in_a_wallet__genshin_impact__,1girl,highres,_seed_1294598571.mp4       |             2.69 |                   1.2  |             2.88 |                      2.7  |                  1.34 |
|  9 | solo,Xiangling,_sweep_dust_with_a_broom__genshin_impact__,1girl,highres,_seed_3483804345.mp4       |             2.72 |                   1.22 |             2.89 |                      2.86 |                  1.2  |
| 25 | solo,Xiangling,_brush_teeth_with_a_toothbrush__genshin_impact__,1girl,highres,_seed_2612536091.mp4 |             2.75 |                   1.23 |             2.91 |                      2.67 |                  1.44 |
| 14 | solo,Xiangling,_store_trash_in_a_bag__genshin_impact__,1girl,highres,_seed_4130052080.mp4          |             2.72 |                   1.25 |             2.86 |                      2.77 |                  1.27 |
'''

import os
import time
import json
import numpy as np
import av
import torch
from PIL import Image
import functools
from transformers import AutoProcessor, AutoConfig
from models.idefics2 import Idefics2ForSequenceClassification
from models.conversation import conv_templates
from typing import List
import csv
import argparse
from tqdm import tqdm
import shutil

# 初始化模型和处理器
processor = AutoProcessor.from_pretrained("TIGER-Lab/VideoScore")
model = Idefics2ForSequenceClassification.from_pretrained("TIGER-Lab/VideoScore", torch_dtype=torch.bfloat16).eval()

MAX_NUM_FRAMES = 24
conv_template = conv_templates["idefics_2"]

VIDEO_EVAL_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos, 
please watch the following frames of a given video and see the text prompt for generating the video, 
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, the consistency of objects or humans in video
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

For each dimension, output a number from [1,2,3,4], 
in which '1' means 'Bad', '2' means 'Average', '3' means 'Good', 
'4' means 'Real' or 'Perfect' (the video is like a real video)
Here is an output example:
visual quality: 4
temporal consistency: 4
dynamic degree: 3
text-to-video alignment: 1
factual consistency: 2

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows: 

"""

aspect_mapping = [
    "visual quality",
    "temporal consistency",
    "dynamic degree",
    "text-to-video alignment",
    "factual consistency",
]

def score(prompt: str, images: List[Image.Image]):
    if not prompt:
        raise ValueError("Please provide a prompt")
    model.to("cuda")
    if not images:
        images = None

    flatten_images = []
    for x in images:
        if isinstance(x, list):
            flatten_images.extend(x)
        else:
            flatten_images.append(x)

    flatten_images = [Image.open(x) if isinstance(x, str) else x for x in flatten_images]
    inputs = processor(text=prompt, images=flatten_images, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    num_aspects = logits.shape[-1]
    aspects = [aspect_mapping[i] for i in range(num_aspects)]

    aspect_scores = {}
    for i, aspect in enumerate(aspects):
        aspect_scores[aspect] = round(logits[0, i].item(), 2)
    return aspect_scores

def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def eval_video(prompt, video: str):
    container = av.open(video)

    total_frames = container.streams.video[0].frames
    if total_frames > MAX_NUM_FRAMES:
        indices = np.arange(0, total_frames, total_frames / MAX_NUM_FRAMES).astype(int)
    else:
        indices = np.arange(total_frames)
    video_frames = read_video_pyav(container, indices)

    frames = [Image.fromarray(x) for x in video_frames]

    eval_prompt = VIDEO_EVAL_PROMPT.format(text_prompt=prompt)

    num_image_token = eval_prompt.count("<image>")
    if num_image_token < len(frames):
        eval_prompt += "<image> " * (len(frames) - num_image_token)

    aspect_scores = score(eval_prompt, [frames])
    return aspect_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate videos in a directory.")
    parser.add_argument("prompt", type=str, help="Text prompt for the video evaluation.")
    parser.add_argument("video_dir", type=str, help="Directory containing video files.")
    args = parser.parse_args()

    video_files = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]

    # 创建五个指标对应的文件夹
    output_dirs = {aspect: f"{aspect}_videos" for aspect in aspect_mapping}
    for dir_name in output_dirs.values():
        os.makedirs(dir_name, exist_ok=True)

    with open("evaluation_results.csv", "w", newline='') as csvfile:
        fieldnames = ["video_name", "visual quality", "temporal consistency", "dynamic degree", "text-to-video alignment", "factual consistency"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for video_file in tqdm(video_files, desc="Evaluating videos"):
            video_name = os.path.basename(video_file)
            aspect_scores = eval_video(args.prompt, video_file)
            aspect_scores["video_name"] = video_name
            writer.writerow(aspect_scores)

            # 将视频文件复制到对应的文件夹中，并以指标值为名称保存
            for aspect, score in aspect_scores.items():
                if aspect != "video_name":
                    score_str = f"{score:.2f}".replace('.', '_')  # 将小数点替换为下划线以便于排序
                    new_video_name = f"{score_str}_{video_name}"
                    output_dir = output_dirs[aspect]
                    shutil.copy(video_file, os.path.join(output_dir, new_video_name))

if __name__ == "__main__":
    main()