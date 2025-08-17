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
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

allImgType = ["." + now_file_type.EXTENSION for now_file_type in FILETYPE_IMAGE]
allImgType.append(".jpeg")

from PIL import Image

# 修改导入方式，避免在主进程中加载模型
def get_worker_function():
    # 延迟导入，在子进程中才加载模型
    from webui.base import worker
    return worker

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
# 提示词数组 [][][][]形式 todo hank 待处理
parser.add_argument("--prompt_arr", type=str, nargs='+', help="Text prompt for video generation")
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
parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video (default: 30)")
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

    # 获取worker函数（在子进程中加载模型）
    worker = get_worker_function()

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
        mp4_crf=now_args.mp4_crf,
        fps=now_args.fps
    )


if __name__ == "__main__":
    if args.source:
        sourceDir = args.source.split(',')
        printMy("sourceDir:", sourceDir)

        # 使用进程池处理多个文件，采用动态提交任务的方式
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # 收集所有待处理的文件
            all_files = []
            for dir in sourceDir:
                printMy("dir:", dir)
                # 深度遍历目录中所有图片
                for root, dirs, files in os.walk(dir):
                    for file in files:
                        # 获取文件名和文件后缀
                        fileName, fileSuffix = os.path.splitext(file)
                        fileSuffix = fileSuffix.lower()
                        if fileSuffix in allImgType:
                            all_files.append(os.path.join(root, file))

            # 动态提交任务
            futures = {}
            # 先提交一部分任务
            initial_count = min(args.max_workers, len(all_files))

            for i in range(initial_count):
                file = all_files[i]
                future = executor.submit(run, args, file, args.prompt, args.seed)
                futures[future] = file

            # 处理剩余的文件
            remaining_files = all_files[initial_count:]

            # 当有任务完成时，提交新任务
            while futures:
                # 等待至少一个任务完成
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)

                # 处理已完成的任务
                for future in done:
                    file = futures.pop(future)
                    try:
                        future.result()
                        printMy(f"Completed processing: {file}")
                    except Exception as e:
                        printMy(f"Error processing {file}: {e}")
                        # 如果进程池损坏，重新创建
                        if "BrokenProcessPool" in str(e):
                            printMy("Process pool is broken, creating a new one...")
                            # 重新提交剩余任务
                            for remaining_file in remaining_files:
                                try:
                                    run(args, remaining_file, args.prompt, args.seed)
                                    printMy(f"Processed {remaining_file} in main process")
                                except Exception as inner_e:
                                    printMy(f"Error processing {remaining_file}: {inner_e}")
                            remaining_files = []
                            break

                # 提交新任务以保持工作池满载
                while remaining_files and len(futures) < args.max_workers:
                    next_file = remaining_files.pop(0)
                    try:
                        future = executor.submit(run, args, next_file, args.prompt, args.seed)
                        futures[future] = next_file
                    except Exception as e:
                        printMy(f"Failed to submit task for {next_file}: {e}")
                        if "BrokenProcessPool" in str(e):
                            # 如果进程池损坏，直接在主进程中处理剩余文件
                            try:
                                run(args, next_file, args.prompt, args.seed)
                                printMy(f"Processed {next_file} in main process after pool failure")
                            except Exception as inner_e:
                                printMy(f"Error processing {next_file} in main process: {inner_e}")

            printMy("All tasks completed")
    else:
        run(args, args.image, args.prompt, args.seed)
