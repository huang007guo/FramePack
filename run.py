# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import traceback
import numpy as np

from diffusers_helper.hf_login import login

import os

import argparse
import numpy as np
from filetype.types import IMAGE as FILETYPE_IMAGE, VIDEO as FILETYPE_VIDEO

# 添加多进程相关导入
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

allImgType = ["." + now_file_type.EXTENSION for now_file_type in FILETYPE_IMAGE]
allImgType.append(".jpeg")

from PIL import Image

from webui.base import worker

def printMy(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    nowDateTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    objects = [nowDateTime, *objects]
    print(*objects, sep=sep, end=end, file=file, flush=flush)
    try:
        print(*objects, sep=sep, end=end, file=open("./log.log", 'a'), flush=flush)
    except BaseException as e:
        print(traceback.format_exc())
        pass

parser = argparse.ArgumentParser()

parser.add_argument("-S", '--source', help="源目录多个用英文逗号分割",
                    nargs='?', type=str, default="")

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
# 添加多进程参数
parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker processes (default: 1)")
args = parser.parse_args()

printMy(args)

# 在文件末尾添加以下代码
def run(now_args, image, prompt="", seed = None):
    # 检查是否提供了输入图像路径
    if image is None:
        raise ValueError("必须提供 --image 参数指定输入图像路径")

    # # 检查是否提供了提示文本
    # if not args.prompt:
    #     raise ValueError("必须提供 --prompt 参数指定提示文本")

    # 加载输入图像
    image = Image.open(image).convert('RGB')
    image = np.array(image)

    # 设置随机种子
    if seed is None:
        seed = np.random.randint(0, 2 ** 31)

    # 调用worker函数
    worker(
        input_image=image,
        prompt=prompt,
        n_prompt=now_args.n_prompt,
        seed=seed,
        total_second_length=now_args.total_second_length,
        latent_window_size=now_args.latent_window_size,
        steps=now_args.steps,
        cfg=now_args.cfg,
        gs=now_args.gs,
        rs=now_args.rs,
        gpu_memory_preservation=now_args.gpu_memory_preservation,
        use_teacache=now_args.use_teacache,
        mp4_crf=now_args.mp4_crf
    )


if __name__ == "__main__":
    if args.source:
        sourceDir = args.source.split(',')
        printMy("sourceDir:", sourceDir)

        # 使用进程池处理多个文件
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for dir in sourceDir:
                printMy("dir:", dir)
                # 深度遍历目录中所有图片
                for root, dirs, files in os.walk(dir):
                    for file in files:
                        # 获取文件名和文件后缀
                        fileName, fileSuffix = os.path.splitext(file)
                        fileSuffix = fileSuffix.lower()
                        if fileSuffix in allImgType:
                            # 提交任务到进程池
                            future = executor.submit(run, args, os.path.join(root, file), args.prompt, args.seed)
                            futures.append(future)

            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    printMy("error:", e)
    else:
        run(args, args.image, args.prompt, args.seed)