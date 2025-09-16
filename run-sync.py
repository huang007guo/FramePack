# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time
import traceback

log_file_path = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), "./log.log")))


def printMy(*objects, sep=' ', end='\n', file=sys.stdout, flush=False):
    nowDateTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    objects = [nowDateTime, *objects]
    print(*objects, sep=sep, end=end, file=file, flush=flush)
    try:
        print(*objects, sep=sep, end=end, file=open(log_file_path, 'a'), flush=flush)
    except BaseException as e:
        print(traceback.format_exc())
        pass


# 输出当前目录,使用管理员运行当前目录会变成: C:\Windows\System32
printMy("当前目录:", os.getcwd())



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
parser.add_argument("--total_second_length", type=float, default=5.0,
                    help="Total video length in seconds (default: 5.0)")
parser.add_argument("--latent_window_size", type=int, default=9, help="Latent window size (default: 9)")
parser.add_argument("--steps", type=int, default=25, help="Number of inference steps (default: 25)")
parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-free guidance scale (default: 1.0)")
parser.add_argument("--gs", type=float, default=10.0, help="Distilled guidance scale (default: 10.0)")
parser.add_argument("--rs", type=float, default=0.0, help="Guidance rescale factor (default: 0.0)")
parser.add_argument("--gpu_memory_preservation", type=float, default=6.0,
                    help="GPU memory preservation in GB (default: 6.0)")
parser.add_argument("--use_teacache", action='store_true', default=True,
                    help="Enable TeaCache optimization (default: True)")
parser.add_argument("--mp4_crf", type=int, default=16, help="MP4 compression quality (lower is better, default: 16)")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for output video (default: 30)")
# 处理完成关机,默认False
parser.add_argument("--shutdown", action='store_true', default=False, help="Shutdown after processing (default: False)")
# 分辨率 resolution 默认 640 只需在调用时传入更大的 resolution 值，如 768 或 1024
parser.add_argument("--resolution", type=int, default=640, help="Resolution for output video (default: 640)")


# 指定时间关机,会完成最后一个后,格式: HH:MM:SS
# parser.add_argument("--shutdown_time", type=str, default=None, help="Specify shutdown time (HH:MM:SS)")
# 最大运行时间 HH,MM,SS 例如 0,0,1; 0,1,0
parser.add_argument("--max_run_time", type=str, default=None, help="Maximum run time (HH,MM,SS)")
# 计划的关机时间,支持当天和隔天
parser.add_argument("--plan_shutdown_time", type=str, default=None, help="Planned shutdown time (HH:MM:SS)")
# 转换完成删除源文件
parser.add_argument("--del_source_file", action='store_true', default=False, help="Delete source file (default: False)")
# 是否只留下最后一个生成的文件,默认False
parser.add_argument("--only_remain_last_file", action='store_true', default=False,
                    help="Only keep the last generated file (default: False)")
# offline_mode
parser.add_argument("--offline_mode", action='store_true', default=False, help="Offline mode (default: False)")

args = parser.parse_args()

printMy(args)
if args.offline_mode:
    os.environ["OFFLINE_MODE"] = '1'

if args.plan_shutdown_time and args.max_run_time is None:
    # 转换为秒数
    plan_shutdown_time_seconds = args.plan_shutdown_time.split(':')
    plan_shutdown_time_seconds = sum([int(x) * 60 ** i for i, x in enumerate(plan_shutdown_time_seconds[::-1])])
    # 当前时间 HH:MM:SS
    current_time = time.strftime("%H:%M:%S", time.localtime())
    current_time_seconds = sum([int(x) * 60 ** i for i, x in enumerate(current_time.split(':')[::-1])])
    # 如果目标时间的秒小于当前代表是第二天
    if plan_shutdown_time_seconds < current_time_seconds:
        plan_shutdown_time_seconds += 24 * 60 * 60
    # 计算差值
    time_difference = plan_shutdown_time_seconds - current_time_seconds
    args.max_run_time = "0,0," + str(time_difference)
    # 输出日志,计算大概需要多少小时多少分钟多少秒
    hours = time_difference // (60 * 60)
    minutes = (time_difference - hours * 60 * 60) // 60
    seconds = time_difference % 60
    printMy("计划关机剩余时间:", f'{hours} 小时 {minutes} 分钟 {seconds} 秒')


if args.max_run_time:
    # 转换为秒数
    args.max_run_time = args.max_run_time.split(',')
    args.max_run_time = sum([int(x) * 60 ** i for i, x in enumerate(args.max_run_time[::-1])])
    printMy("max_run_time:", args.max_run_time)
# 开始运行时间
start_time = time.time()

import numpy as np
from filetype.types import IMAGE as FILETYPE_IMAGE

# 添加多进程相关导入

allImgType = ["." + now_file_type.EXTENSION for now_file_type in FILETYPE_IMAGE]
allImgType.append(".jpeg")

from PIL import Image
from base import worker

# 在文件末尾添加以下代码
def run(now_args, image, prompt="", seed=None, file_name=None):
    # 检查是否提供了输入图像路径
    if image is None:
        raise ValueError("必须提供 --image 参数指定输入图像路径")
    # 如果图片是路径识别图片名称
    if file_name is None and type(image) == str:
        file_name, file_suffix = os.path.splitext(os.path.basename(image))
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
        mp4_crf=now_args.mp4_crf,
        fps=now_args.fps,
        resolution=now_args.resolution,
        file_name=file_name,
        only_remain_last_file=now_args.only_remain_last_file,
    )


def main():
    if args.source:
        sourceDir = args.source.split(',')
        printMy("sourceDir:", sourceDir)

        # 使用进程池处理多个文件，采用动态提交任务的方式
        # 收集所有待处理的文件
        all_files = []
        for dir in sourceDir:
            printMy("dir:", dir)
            # 深度遍历目录中所有图片
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if args.max_run_time and time.time() - start_time > args.max_run_time:
                        printMy("已到最大运行时间,结束处理!")
                        if args.shutdown:
                            os.system("shutdown -a")
                            os.system("shutdown -s -t 120")
                        return
                    # 获取文件名和文件后缀
                    fileName, fileSuffix = os.path.splitext(file)
                    fileSuffix = fileSuffix.lower()
                    if fileSuffix in allImgType:
                        try:
                            # 获取图片路径
                            filePath = os.path.join(root, file)
                            printMy("filePath:", filePath)
                            run(args, filePath, args.prompt, args.seed)
                            printMy("处理完成:", filePath)
                            if args.del_source_file:
                                os.remove(filePath)
                        except Exception as e:
                            printMy("error:", e)
    else:
        run(args, args.image, args.prompt, args.seed)
    # 处理完成关机
    if args.shutdown:
        os.system("shutdown -a")
        os.system("shutdown -s -t 120")


if __name__ == "__main__":
    main()
