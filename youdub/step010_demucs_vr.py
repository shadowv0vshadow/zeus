"""Demucs 音频分离模块

该模块提供基于 Demucs 的音频分离功能，用于分离人声和伴奏
"""

import os
import time
from typing import Optional

import torch
from demucs.api import Separator
from loguru import logger

from .utils import save_wav

# 自动选择设备
AUTO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局模型缓存
separator: Optional[Separator] = None


def init_demucs() -> None:
    """初始化 Demucs 模型"""
    global separator
    if separator is None:
        load_model()


def load_model(model_name: str = "htdemucs_ft", device: str = "auto", progress: bool = True, shifts: int = 5) -> None:
    """加载 Demucs 模型

    Args:
        model_name: 模型名称
        device: 设备选择，'auto'/'cuda'/'cpu'
        progress: 是否显示进度条
        shifts: Shifts 参数，增加可以提高质量但会更慢
    """
    global separator

    if separator is not None:
        logger.info("Demucs model already loaded")
        return

    logger.info(f"Loading Demucs model: {model_name}")
    t_start = time.time()

    # 选择设备
    if device == "auto":
        device_obj = AUTO_DEVICE
    else:
        device_obj = torch.device(device)

    separator = Separator(model_name, device=device_obj, progress=progress, shifts=shifts)

    t_end = time.time()
    logger.info(f"Demucs model loaded in {t_end - t_start:.2f} seconds")


def separate_audio(
    folder: str, model_name: str = "htdemucs_ft", device: str = "auto", progress: bool = True, shifts: int = 5
) -> None:
    """分离音频中的人声和伴奏

    Args:
        folder: 音频文件所在文件夹
        model_name: Demucs 模型名称
        device: 设备选择
        progress: 是否显示进度条
        shifts: Shifts 参数
    """
    global separator

    audio_path = os.path.join(folder, "audio.wav")
    if not os.path.exists(audio_path):
        logger.warning(f"Audio file not found: {audio_path}")
        return

    vocal_output_path = os.path.join(folder, "audio_vocals.wav")
    instruments_output_path = os.path.join(folder, "audio_instruments.wav")

    # 检查是否已经分离
    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f"Audio already separated in {folder}")
        return

    logger.info(f"Separating audio from {folder}")

    # 加载模型
    load_model(model_name, device, progress, shifts)

    # 分离音频
    t_start = time.time()
    try:
        origin, separated = separator.separate_audio_file(audio_path)
    except Exception as e:
        logger.error(f"Error separating audio from {folder}: {e}")
        raise Exception(f"Failed to separate audio: {e}")

    t_end = time.time()
    logger.info(f"Audio separated in {t_end - t_start:.2f} seconds")

    # 提取人声
    vocals = separated["vocals"].numpy().T

    # 合并所有非人声轨道作为伴奏
    instruments = None
    for key, value in separated.items():
        if key == "vocals":
            continue
        if instruments is None:
            instruments = value
        else:
            instruments += value

    instruments = instruments.numpy().T

    # 保存分离后的音频
    save_wav(vocals, vocal_output_path, sample_rate=44100)
    logger.info(f"Vocals saved to {vocal_output_path}")

    save_wav(instruments, instruments_output_path, sample_rate=44100)
    logger.info(f"Instruments saved to {instruments_output_path}")


def extract_audio_from_video(folder: str) -> bool:
    """从视频中提取音频

    Args:
        folder: 视频文件所在文件夹

    Returns:
        是否提取成功
    """
    video_path = os.path.join(folder, "download.mp4")
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return False

    audio_path = os.path.join(folder, "audio.wav")
    if os.path.exists(audio_path):
        logger.info(f"Audio already extracted in {folder}")
        return True

    logger.info(f"Extracting audio from {folder}")

    # 使用 ffmpeg 提取音频
    ffmpeg_command = (
        f'ffmpeg -loglevel error -i "{video_path}" ' f'-vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"'
    )

    return_code = os.system(ffmpeg_command)

    if return_code != 0:
        logger.error(f"Failed to extract audio from {video_path}")
        return False

    time.sleep(1)
    logger.info(f"Audio extracted to {audio_path}")
    return True


def separate_all_audio_under_folder(
    root_folder: str, model_name: str = "htdemucs_ft", device: str = "auto", progress: bool = True, shifts: int = 5
) -> str:
    """分离指定文件夹下所有视频的音频

    Args:
        root_folder: 根文件夹路径
        model_name: Demucs 模型名称
        device: 设备选择
        progress: 是否显示进度条
        shifts: Shifts 参数

    Returns:
        处理结果描述
    """
    logger.info(f"Starting audio separation for all files under: {root_folder}")
    processed_count = 0

    for subdir, dirs, files in os.walk(root_folder):
        if "download.mp4" not in files:
            continue

        # 提取音频
        if "audio.wav" not in files:
            if not extract_audio_from_video(subdir):
                continue

        # 分离音频
        if "audio_vocals.wav" not in files:
            try:
                separate_audio(subdir, model_name, device, progress, shifts)
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to separate audio in {subdir}: {e}")

    result_msg = f"Separated {processed_count} audio files under {root_folder}"
    logger.info(result_msg)
    return result_msg


if __name__ == "__main__":
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)
