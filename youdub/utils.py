"""工具函数模块

提供通用的工具函数，包括：
- 文件名清理
- 音频文件读写
- 音频归一化处理
"""

import re
import string
import numpy as np
from scipy.io import wavfile


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符

    Args:
        filename: 原始文件名

    Returns:
        清理后的文件名
    """
    # 定义合法字符集
    valid_chars = f"-_.() {string.ascii_letters}{string.digits}"

    # 只保留合法字符
    sanitized_filename = "".join(c for c in filename if c in valid_chars)

    # 将多个连续空格替换为单个空格
    sanitized_filename = re.sub(r" +", " ", sanitized_filename)

    return sanitized_filename.strip()


def save_wav(wav: np.ndarray, output_path: str, sample_rate: int = 24000) -> None:
    """保存音频为 WAV 文件（不进行归一化）

    Args:
        wav: 音频数据（numpy 数组）
        output_path: 输出文件路径
        sample_rate: 采样率，默认 24000
    """
    # 直接缩放到 16 位整数范围
    wav_norm = wav * 32767
    wav_int16 = np.clip(wav_norm, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, wav_int16)


def save_wav_norm(wav: np.ndarray, output_path: str, sample_rate: int = 24000) -> None:
    """保存音频为 WAV 文件（进行归一化）

    将音频归一化到最大振幅后保存

    Args:
        wav: 音频数据（numpy 数组）
        output_path: 输出文件路径
        sample_rate: 采样率，默认 24000
    """
    max_amplitude = max(0.01, np.max(np.abs(wav)))
    wav_norm = wav * (32767 / max_amplitude)
    wav_int16 = np.clip(wav_norm, -32768, 32767).astype(np.int16)
    wavfile.write(output_path, sample_rate, wav_int16)


def normalize_wav(wav_path: str) -> None:
    """归一化音频文件并覆盖原文件

    Args:
        wav_path: WAV 文件路径
    """
    sample_rate, wav = wavfile.read(wav_path)
    max_amplitude = max(0.01, np.max(np.abs(wav)))
    wav_norm = wav * (32767 / max_amplitude)
    wav_int16 = np.clip(wav_norm, -32768, 32767).astype(np.int16)
    wavfile.write(wav_path, sample_rate, wav_int16)
