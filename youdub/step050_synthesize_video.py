# -*- coding: utf-8 -*-
"""视频合成模块

该模块提供视频合成功能，包括：
- 生成字幕文件（SRT格式）
- 合成视频、音频和字幕
- 视频处理和格式转换
"""

import json
import os
import subprocess
import time
from typing import List, Dict, Any, Tuple

from loguru import logger


def split_text(input_data: List[Dict[str, Any]], punctuations: List[str] = None) -> List[Dict[str, Any]]:
    """将翻译文本按标点符号分割成短句

    Args:
        input_data: 包含翻译信息的字典列表
        punctuations: 用于分句的标点符号列表

    Returns:
        分割后的翻译数据列表
    """
    if punctuations is None:
        punctuations = ["，", "；", "：", "。", "？", "！", "\n", '"']

    def is_punctuation(char: str) -> bool:
        """检查字符是否为标点符号"""
        return char in punctuations

    output_data = []

    for item in input_data:
        start = item["start"]
        text = item["translation"]
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = item["text"]
        sentence_start = 0

        # 计算每个字符的时长
        if len(text) == 0:
            continue

        duration_per_char = (item["end"] - item["start"]) / len(text)

        for i, char in enumerate(text):
            # 检查是否需要分句
            is_last_char = i == len(text) - 1
            is_too_short = i - sentence_start < 5
            next_is_punct = i < len(text) - 1 and is_punctuation(text[i + 1])

            # 如果不是标点符号且不是最后一个字符，继续
            if not is_punctuation(char) and not is_last_char:
                continue

            # 如果句子太短且不是最后一个字符，继续
            if is_too_short and not is_last_char:
                continue

            # 如果下一个字符也是标点符号，继续
            if next_is_punct:
                continue

            # 分割句子
            sentence = text[sentence_start : i + 1]
            sentence_end = start + duration_per_char * len(sentence)

            output_data.append(
                {
                    "start": round(start, 3),
                    "end": round(sentence_end, 3),
                    "text": original_text,
                    "translation": sentence,
                    "speaker": speaker,
                }
            )

            # 更新下一句的开始位置
            start = sentence_end
            sentence_start = i + 1

    return output_data


def format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间格式

    Args:
        seconds: 秒数

    Returns:
        SRT 格式的时间字符串 (HH:MM:SS,mmm)
    """
    millisec = int((seconds - int(seconds)) * 1000)
    hours, remaining_seconds = divmod(int(seconds), 3600)
    minutes, secs = divmod(remaining_seconds, 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{millisec:03}"


def generate_srt(
    translation: List[Dict[str, Any]], srt_path: str, speed_up: float = 1.0, max_line_char: int = 30
) -> None:
    """生成 SRT 字幕文件

    Args:
        translation: 翻译数据列表
        srt_path: 输出的 SRT 文件路径
        speed_up: 速度倍率
        max_line_char: 每行最大字符数
    """
    translation = split_text(translation)

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(translation):
            start = format_timestamp(item["start"] / speed_up)
            end = format_timestamp(item["end"] / speed_up)
            text = item["translation"]

            # 去除末尾的所有标点符号（包括中文和英文标点）
            # 包括：句号、问号、感叹号、逗号、分号、冒号、引号等
            punctuation_chars = (
                "。.！!？?，,；;：:、"  # 基本标点
                + '"'  # 英文双引号
                + "'"  # 英文单引号
                + '"'  # 中文左双引号
                + "'"  # 中文左单引号
                + '"'  # 中文右双引号
                + "'"  # 中文右单引号
            )
            text = text.rstrip(punctuation_chars)

            # 如果文本为空，跳过
            if not text.strip():
                continue

            # 计算需要分成几行
            num_lines = len(text) // (max_line_char + 1) + 1
            chars_per_line = min(round(len(text) / num_lines), max_line_char)

            # 按行分割文本
            text_lines = [text[j * chars_per_line : (j + 1) * chars_per_line] for j in range(num_lines)]
            # 确保每行末尾也不带标点符号
            text_lines = [line.rstrip(punctuation_chars) for line in text_lines]
            # 过滤掉空行
            text_lines = [line for line in text_lines if line.strip()]
            formatted_text = "\n".join(text_lines)
            
            # 如果所有行都被过滤掉了，跳过这条字幕
            if not formatted_text.strip():
                continue

            # 写入 SRT 格式
            f.write(f"{i + 1}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{formatted_text}\n\n")


def get_aspect_ratio(video_path: str) -> float:
    """获取视频的宽高比

    Args:
        video_path: 视频文件路径

    Returns:
        宽高比（宽度/高度）
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        video_path,
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    if not data.get("streams"):
        raise ValueError(f"No video stream found in {video_path}")

    dimensions = data["streams"][0]
    width = dimensions["width"]
    height = dimensions["height"]

    if height == 0:
        raise ValueError(f"Invalid video dimensions: {width}x{height}")

    return width / height


def convert_resolution(aspect_ratio: float, resolution: str = "1080p") -> Tuple[int, int]:
    """根据宽高比和目标分辨率计算视频尺寸

    Args:
        aspect_ratio: 宽高比
        resolution: 目标分辨率（如 '1080p', '720p'）

    Returns:
        (宽度, 高度) 元组
    """
    target_pixels = int(resolution[:-1])

    if aspect_ratio < 1:
        # 竖屏视频
        width = target_pixels
        height = int(width / aspect_ratio)
    else:
        # 横屏视频
        height = target_pixels
        width = int(height * aspect_ratio)

    # 确保宽高都是偶数（ffmpeg 要求）
    width = width - width % 2
    height = height - height % 2

    return width, height


def synthesize_video(
    folder: str, subtitles: bool = True, speed_up: float = 1.0, fps: int = 30, resolution: str = "1080p"
) -> None:
    """合成最终视频

    Args:
        folder: 工作文件夹路径
        subtitles: 是否添加字幕
        speed_up: 视频加速倍率
        fps: 目标帧率
        resolution: 目标分辨率
    """
    output_video = os.path.join(folder, "video.mp4")

    if os.path.exists(output_video):
        logger.info(f"Video already synthesized in {folder}")
        return

    # 检查必需的输入文件
    translation_path = os.path.join(folder, "translation.json")
    input_audio = os.path.join(folder, "audio_combined.wav")
    input_video = os.path.join(folder, "download.mp4")

    if not os.path.exists(translation_path):
        logger.warning(f"Translation file not found: {translation_path}")
        return

    if not os.path.exists(input_audio):
        logger.warning(f"Audio file not found: {input_audio}")
        return

    if not os.path.exists(input_video):
        logger.warning(f"Video file not found: {input_video}")
        return

    logger.info(f"Synthesizing video in {folder}")

    # 加载翻译数据
    with open(translation_path, "r", encoding="utf-8") as f:
        translation = json.load(f)

    # 生成字幕文件
    srt_path = os.path.join(folder, "subtitles.srt")
    generate_srt(translation, srt_path, speed_up)

    # 处理 Windows 路径分隔符
    srt_path = srt_path.replace("\\", "/")

    # 计算视频尺寸
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    resolution_str = f"{width}x{height}"

    # 计算字体大小和轮廓
    font_size = int(width / 128)
    outline = int(round(font_size / 8))

    # 构建 FFmpeg 滤镜
    video_speed_filter = f"setpts=PTS/{speed_up}"
    audio_speed_filter = f"atempo={speed_up}"

    if subtitles:
        subtitle_filter = (
            f"subtitles={srt_path}:"
            f"force_style='FontName=Songti SC,"
            f"FontSize={font_size},"
            f"PrimaryColour=&HFFFFFF,"
            f"OutlineColour=&H000000,"
            f"Outline={outline},"
            f"WrapStyle=2'"
        )
        filter_complex = f"[0:v]{video_speed_filter},{subtitle_filter}[v];[1:a]{audio_speed_filter}[a]"
    else:
        filter_complex = f"[0:v]{video_speed_filter}[v];[1:a]{audio_speed_filter}[a]"

    # 构建 FFmpeg 命令
    ffmpeg_command = [
        "ffmpeg",
        "-i",
        input_video,
        "-i",
        input_audio,
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-r",
        str(fps),
        "-s",
        resolution_str,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        output_video,
        "-y",
    ]

    logger.info(f'Running FFmpeg command: {" ".join(ffmpeg_command[:5])}...')
    try:
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        if result.stderr:
            logger.debug(f"FFmpeg output: {result.stderr[-500:]}")  # 只显示最后500字符
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed with exit code {e.returncode}")
        logger.error(f'FFmpeg error: {e.stderr[:1000] if e.stderr else "No error output"}')
        raise

    time.sleep(1)
    logger.info(f"Video synthesized successfully: {output_video}")


def synthesize_all_video_under_folder(
    folder: str, subtitles: bool = True, speed_up: float = 1.0, fps: int = 30, resolution: str = "1080p"
) -> str:
    """合成指定文件夹下的所有视频

    Args:
        folder: 根文件夹路径
        subtitles: 是否添加字幕
        speed_up: 视频加速倍率
        fps: 目标帧率
        resolution: 目标分辨率

    Returns:
        处理结果描述
    """
    logger.info(f"Starting video synthesis for all videos under: {folder}")
    synthesized_count = 0

    for root, dirs, files in os.walk(folder):
        if "download.mp4" in files and "video.mp4" not in files:
            try:
                synthesize_video(root, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=resolution)
                synthesized_count += 1
            except Exception as e:
                logger.error(f"Error synthesizing video in {root}: {e}")

    result_msg = f"Synthesized {synthesized_count} videos under {folder}"
    logger.info(result_msg)
    return result_msg


if __name__ == "__main__":
    folder = r"videos\3Blue1Brown\20231207 Im still astounded this is true"
    synthesize_all_video_under_folder(folder, subtitles=True)
