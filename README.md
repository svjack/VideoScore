# VideoScore
This is the official repo for our EMNLP 2024 paper "VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation".

<a target="_blank" href="https://arxiv.org/abs/2406.15252">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://github.com/TIGER-AI-Lab/VideoScore">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-green?style=flat&logo=github"></a>
<a target="_blank" href="https://tiger-ai-lab.github.io/VideoScore/">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20Website-blue?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/VideoFeedback">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/VideoScore">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Demo-red?style=flat"></a> 
<a target="_blank" href="https://huggingface.co/TIGER-Lab/VideoScore">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Models-red?style=flat"></a>
<a target="_blank" href="https://twitter.com/DongfuJiang/status/1805438506137010326">
<img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a>
<br>

# VideoScore

## 概述
**VideoScore** 项目旨在从多个维度评估视频质量。它提供了对视觉质量、时间一致性、动态程度、文本与视频对齐以及事实一致性的全面评估。该项目使用 Python 构建，并利用各种库来分析和评分视频内容。

## 入门指南

### 前提条件
在开始之前，请确保您已安装以下软件：
- Python 3.x
- Git

### 安装
1. 克隆仓库：
   ```bash
   git clone https://huggingface.co/spaces/svjack/VideoScore && cd VideoScore
   ```

2. 安装所需的依赖项：
   ```bash
   pip install -r requirements.txt
   ```

## 维度映射
以下是 `aspect_mapping` 中各个词汇的涵义：

1. **视觉质量**:
   - **涵义**: 视觉质量
   - **解释**: 视频在清晰度、分辨率、亮度和色彩等方面的质量。这个维度评估视频的视觉表现，包括图像的清晰度、色彩的准确性和整体的视觉吸引力。

2. **时间一致性**:
   - **涵义**: 时间一致性
   - **解释**: 视频中物体或人物的一致性。这个维度评估视频在时间上的连贯性，即视频中的物体或人物在不同帧之间是否保持一致，没有明显的跳跃或不连贯的现象。

3. **动态程度**:
   - **涵义**: 动态程度
   - **解释**: 视频中动态变化的程度。这个维度评估视频的动态性，即视频中物体或场景的变化程度，包括运动的频率和幅度。

4. **文本与视频对齐**:
   - **涵义**: 文本与视频的对齐
   - **解释**: 文本提示与视频内容之间的对齐程度。这个维度评估视频内容与给定文本提示之间的匹配程度，即视频是否准确地反映了文本提示所描述的内容。

5. **事实一致性**:
   - **涵义**: 事实一致性
   - **解释**: 视频内容与常识和事实知识的一致性。这个维度评估视频内容是否符合常识和事实知识，即视频中的内容是否真实可信，没有明显的逻辑错误或与现实不符的情况。

这些维度共同构成了对视频质量的多方面评估，涵盖了从视觉表现到内容一致性的各个方面。

## 示例
- APP
```bash
python app_regression.py
```
Or
- 脚本
以下是一个使用 `video_regression_script.py` 的示例：
```bash
python video_regression_script.py "" xiangling_mp4_dir_tiny
```

### 结果分析
使用 Pandas 读取并分析评估结果：

```python
import pandas as pd
edf = pd.read_csv("evaluation_results.csv")
edf.describe()
print(edf.sort_values(by = "temporal consistency", ascending = True).head(5).to_markdown())
```

输出结果：

|    | video_name                                                                                         |   visual quality |   temporal consistency |   dynamic degree |   text-to-video alignment |   factual consistency |
|---:|:---------------------------------------------------------------------------------------------------|-----------------:|-----------------------:|-----------------:|--------------------------:|----------------------:|
| 26 | solo,Xiangling,_shave_with_a_razor__genshin_impact__,1girl,highres,_seed_3140464511.mp4            |             2.8  |                   1.14 |             2.97 |                      2.78 |                  1.26 |
|  0 | solo,Xiangling,_carry_money_in_a_wallet__genshin_impact__,1girl,highres,_seed_1294598571.mp4       |             2.69 |                   1.2  |             2.88 |                      2.7  |                  1.34 |
|  9 | solo,Xiangling,_sweep_dust_with_a_broom__genshin_impact__,1girl,highres,_seed_3483804345.mp4       |             2.72 |                   1.22 |             2.89 |                      2.86 |                  1.2  |
| 25 | solo,Xiangling,_brush_teeth_with_a_toothbrush__genshin_impact__,1girl,highres,_seed_2612536091.mp4 |             2.75 |                   1.23 |             2.91 |                      2.67 |                  1.44 |
| 14 | solo,Xiangling,_store_trash_in_a_bag__genshin_impact__,1girl,highres,_seed_4130052080.mp4          |             2.72 |                   1.25 |             2.86 |                      2.77 |                  1.27 |

## Make video dataset in svjack/Genshin-Impact-XiangLing-animatediff-with-score-organized
```python
import os
import shutil
import uuid
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条

# 读取 evaluation_results.csv 文件
vdf = pd.read_csv("xiangling_benchmark_dir/evaluation_results.csv")

# 将 video_name 列的路径转换为完整路径
vdf["video_name"] = vdf["video_name"].map(lambda x: os.path.join("xiangling_benchmark_dir/xiangling_mp4_dir_total/", x))

def process_files(input_path, output_path, prefix=""):
    # 创建输出路径
    os.makedirs(output_path, exist_ok=True)

    # 获取所有 mp4 文件
    mp4_files = list(Path(input_path).rglob("*.mp4"))

    # 创建 metadata 列表
    metadata = []

    # 使用 tqdm 显示进度条
    for mp4_file in tqdm(mp4_files, desc="Processing files"):
        # 生成 UUID
        unique_id = str(uuid.uuid4())

        # 构建新的文件名
        new_mp4_file = Path(output_path) / f"{unique_id}.mp4"
        new_txt_file = Path(output_path) / f"{unique_id}.txt"

        # 拷贝 mp4 文件到新路径并重命名
        shutil.copy(mp4_file, new_mp4_file)

        # 解析视频路径，生成 prompt
        video_name = Path(mp4_file).name  # 获取文件名
        video_name_without_seed = video_name.split("_seed_")[0]  # 去掉 _seed_xxxxxx.mp4 部分
        video_name_without_seed = video_name_without_seed.replace("_", " ")  # 将下划线替换为空格

        # 在内容前添加 prefix
        modified_prompt = f"{prefix}{video_name_without_seed}"

        # 将修改后的内容写入新的 txt 文件
        with open(new_txt_file, "w", encoding="utf-8") as f:
            f.write(modified_prompt)

        # 从 vdf 中提取相关信息
        row = vdf[vdf["video_name"] == str(mp4_file)]
        if not row.empty:
            visual_quality = row["visual quality"].values[0]
            temporal_consistency = row["temporal consistency"].values[0]
            dynamic_degree = row["dynamic degree"].values[0]
            factual_consistency = row["factual consistency"].values[0]

            # 添加到 metadata 列表
            metadata.append({
                "file_name": f"{unique_id}.mp4",
                "prompt": modified_prompt,
                "visual quality": visual_quality,
                "temporal consistency": temporal_consistency,
                "dynamic degree": dynamic_degree,
                "factual consistency": factual_consistency,
                "original_file_name": video_name,  # 添加重命名前的文件名
            })

    # 生成 metadata.csv 文件
    df = pd.DataFrame(metadata)
    df.to_csv(Path(output_path) / "metadata.csv", index=False)

# 示例调用
input_path = "xiangling_benchmark_dir/xiangling_mp4_dir_total"
output_path = "xiangling_benchmark_dir/xiangling_processed"
process_files(input_path, output_path)
```

## And upload by (pip install huggingface_hub)
```bash
huggingface-cli upload svjack/Genshin-Impact-XiangLing-animatediff-with-score-organized xiangling_processed/* . --repo-type dataset
```

## News
[2024-11-28] Try on our new version [VideoScore-v1.1](https://huggingface.co/TIGER-Lab/VideoScore-v1.1), with better performance in **"text-to-video alignment"** subscore and the support for **48 frames** in inference now!

[2024-08-05] We released the Wandb training cruves of [VideoScore](https://api.wandb.ai/links/xuanhe/ptohlfcx) and [VideoScore-anno-only](https://api.wandb.ai/links/xuanhe/4vs5k0cq) to help reproduce the training results.


## Introduction

<video src="https://user-images.githubusercontent.com/105091430/90adfb70-fdff-4101-9207-9bd4f43aae4c.mp4"></video>


🚀The recent years have witnessed great advances in video generation. However, the development of automatic video metrics is lagging significantly behind. None of the existing metric is able to provide reliable scores over generated videos. 
🤔The main barrier is the lack of large-scale human-annotated dataset.

- 🛢️**VideoFeedback Dataset**. In this paper, we release VideoFeedback, the first large-scale dataset containing human-provided multiaspect score over 37.6K synthesized videos from 11 existing video generative models.

- 🏅**VideoScore**. We train VideoScore (initialized from Mantis) based on VideoFeedback to enable automatic video quality assessment. Experiments show that the Spearman correlation between VideoScore and humans can reach 77.1 on VideoFeedback-test, beating the prior best metrics by about 50 points. Further result on other held-out EvalCrafter, GenAI-Bench, and VBench show that VideoScore has consistently much higher correlation with human judges than other metrics.

- 🫡**Human Feedback for Video generative models**. Due to these results, we believe VideoScore can serve as a great proxy for human raters to (1) rate different video models to track progress (2) simulate fine-grained human feedback in Reinforcement Learning with Human Feedback (RLHF) to improve current video generation models.

## Installation

- for inference
```bash
pip install -e . 
```
- for evaluation
```bash
pip install -e .[eval] 
```
- for training
```bash
git clone https://github.com/TIGER-AI-Lab/Mantis
cd Mantis
pip install -e .[train,eval]
pip install flash-attn --no-build-isolation
# then training scripts are in Mantis/train/scripts
```

## Dataset
- [🤗 VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback) VideoFeedback contains a total of 37.6K text-to-video pairs from 11 popular video generative models, with some real-world videos as data augmentation. The videos are annotated by raters for five evaluation dimensions: Visual Quality, Temporal Consistency, Dynamic Degree, Text-to-Video Alignment and Factual Consistency, in 1-4 scoring scale. 

- [🤗 VideoScore-Bench](https://huggingface.co/datasets/TIGER-Lab/VideoScore-Bench) 
We derive four test sets from 
[VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback), 
[EvalCrafter](https://github.com/evalcrafter/EvalCrafter), 
[GenAI-Bench](https://huggingface.co/datasets/TIGER-Lab/GenAI-Bench) and 
[VBench](https://github.com/Vchitect/VBench) respectively to curate VideoScore-Bench. 
VideoScore-Bench is composed of about 7,000 videos, covering both Likert-scale annotation and human preference data.  

## Model
- [🤗 VideoScore](https://huggingface.co/TIGER-Lab/VideoScore) is a video quality evaluation model, taking [Mantis-8B-Idefics2](https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2) as base-model and trained on [VideoFeedback](https://huggingface.co/datasets/TIGER-Lab/VideoFeedback). 

- [🤗 VideoScore-anno-only](https://huggingface.co/TIGER-Lab/VideoScore-anno-only) is a variant of VideoScore, trained on VideoFeedback with the real videos excluded.


## Inference examples
```bash
cd examples
python run_videoscore.py
```

## Evaluation
For details, please check [benchmark/README.md](benchmark/README.md)

## Training
For details, please check [training/README.md](training/README.md)

## Acknowledgement
- Thanks [Mantis](https://github.com/TIGER-AI-Lab/Mantis/tree/main) for the training codebase of VideoScore (and variants) and also for the plug-and-play MLLM tools in evaluation stage! 

- Thanks [VIEScore](https://github.com/TIGER-AI-Lab/VIEScore/tree/main) for some codes of prompting MLLM in evaluation! 

## Citation
```bibtex
@article{he2024videoscore,
  title = {VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation},
  author = {He, Xuan and Jiang, Dongfu and Zhang, Ge and Ku, Max and Soni, Achint and Siu, Sherman and Chen, Haonan and Chandra, Abhranil and Jiang, Ziyan and Arulraj, Aaran and Wang, Kai and Do, Quy Duc and Ni, Yuansheng and Lyu, Bohan and Narsupalli, Yaswanth and Fan, Rongqi and Lyu, Zhiheng and Lin, Yuchen and Chen, Wenhu},
  journal = {ArXiv},
  year = {2024},
  volume={abs/2406.15252},
  url = {https://arxiv.org/abs/2406.15252},
}

```
