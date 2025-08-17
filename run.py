import numpy as np

from diffusers_helper.hf_login import login

import os

import argparse

from PIL import Image

from webui.base import worker

parser = argparse.ArgumentParser()

# 添加视频生成相关参数
parser.add_argument("--image", type=str, help="Path to input image")
parser.add_argument("--prompt", type=str, default="", help="Text prompt for video generation")
parser.add_argument("--n_prompt", type=str, default="", help="Negative prompt (default: '')")
parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 31337)")
parser.add_argument("--total_second_length", type=float, default=5.0, help="Total video length in seconds (default: 5.0)")
parser.add_argument("--latent_window_size", type=int, default=9, help="Latent window size (default: 9)")
parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (default: 25)")
parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-free guidance scale (default: 1.0)")
parser.add_argument("--gs", type=float, default=10.0, help="Distilled guidance scale (default: 10.0)")
parser.add_argument("--rs", type=float, default=0.0, help="Guidance rescale factor (default: 0.0)")
parser.add_argument("--gpu_memory_preservation", type=float, default=6.0, help="GPU memory preservation in GB (default: 6.0)")
parser.add_argument("--use_teacache", action='store_true', default=True, help="Enable TeaCache optimization (default: True)")
parser.add_argument("--mp4_crf", type=int, default=16, help="MP4 compression quality (lower is better, default: 16)")
args = parser.parse_args()

print(args)

# 在文件末尾添加以下代码
if __name__ == "__main__":
    # 检查是否提供了输入图像路径
    if args.image is None:
        raise ValueError("必须提供 --image 参数指定输入图像路径")

    # # 检查是否提供了提示文本
    # if not args.prompt:
    #     raise ValueError("必须提供 --prompt 参数指定提示文本")

    # 加载输入图像
    image = Image.open(args.image).convert('RGB')
    image = np.array(image)

    # 设置随机种子
    if args.seed is None:
        args.seed = np.random.randint(0, 2**31)

    # 调用worker函数
    worker(
        input_image=image,
        prompt=args.prompt,
        n_prompt=args.n_prompt,
        seed=args.seed,
        total_second_length=args.total_second_length,
        latent_window_size=args.latent_window_size,
        steps=args.steps,
        cfg=args.cfg,
        gs=args.gs,
        rs=args.rs,
        gpu_memory_preservation=args.gpu_memory_preservation,
        use_teacache=args.use_teacache,
        mp4_crf=args.mp4_crf
    )
