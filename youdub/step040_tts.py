"""TTS（文本转语音）模块

该模块提供文本转语音功能，支持：
- 字节跳动 TTS
- XTTS 模型
- 音频时长调整
"""

import json
import os
import re
from typing import List, Dict, Any

import librosa
import numpy as np
from audiostretchy.stretch import stretch_audio
from loguru import logger

from .cn_tx import TextNorm
from .step041_tts_bytedance import tts as bytedance_tts
from .step042_tts_xtts import tts as xtts_tts
from .step043_tts_aliyun import tts as aliyun_tts
from .utils import save_wav, save_wav_norm

# 初始化文本规范化器
normalizer = TextNorm()


def preprocess_text(text: str) -> str:
    """预处理文本，进行规范化
    
    Args:
        text: 原始文本
        
    Returns:
        处理后的文本
    """
    # 替换特殊词汇
    text = text.replace('AI', '人工智能')
    
    # 在大写字母前插入空格（除了开头）
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    
    # 文本规范化
    text = normalizer(text)
    
    # 在字母和数字之间插入空格
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    
    return text


def adjust_audio_length(
    wav_path: str,
    desired_length: float,
    sample_rate: int = 24000,
    min_speed_factor: float = 0.6,
    max_speed_factor: float = 1.1
) -> tuple:
    """调整音频长度以匹配目标时长
    
    Args:
        wav_path: 音频文件路径
        desired_length: 目标时长（秒）
        sample_rate: 采样率
        min_speed_factor: 最小速度因子
        max_speed_factor: 最大速度因子
        
    Returns:
        (调整后的音频数据, 实际时长)
    """
    wav, sample_rate = librosa.load(wav_path, sr=sample_rate)
    current_length = len(wav) / sample_rate
    
    # 计算速度因子，限制在合理范围内
    speed_factor = desired_length / current_length
    speed_factor = max(min(speed_factor, max_speed_factor), min_speed_factor)
    
    actual_length = current_length * speed_factor
    target_path = wav_path.replace('.wav', '_adjusted.wav')
    
    # 拉伸音频
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    
    # 重新加载调整后的音频
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)
    
    # 截取到目标长度
    target_samples = int(actual_length * sample_rate)
    return wav[:target_samples], actual_length


def generate_wavs(folder: str, tts_model: str = 'xtts_v2', xtts_model_name: str = 'tts_models/multilingual/multi-dataset/xtts_v2', aliyun_voice: str = None) -> None:
    """生成所有片段的语音
    
        Args:
        folder: 工作文件夹路径
        tts_model: TTS 模型选择，可选值：
            - 'bytedance': 字节跳动 TTS
            - 'aliyun': 阿里云 TTS
            - 'xtts_v1': XTTS v1
            - 'xtts_v2': XTTS v2 (默认)
            - 'xtts_v2.0.2': XTTS v2.0.2
        xtts_model_name: XTTS 模型路径（仅当 tts_model 为 xtts_* 时有效）
        aliyun_voice: 阿里云 TTS 音色名称（仅当 tts_model 为 aliyun 时有效）
    """
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    os.makedirs(output_folder, exist_ok=True)
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    speakers = set()

    for line in transcript:
        speakers.add(line['speaker'])
    num_speakers = len(speakers)
    logger.info(f'Found {num_speakers} speakers')

    full_wav = np.zeros((0, ))
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')

        print(output_path)
        print(speaker_wav)
        
        # 根据选择的 TTS 模型调用对应的 TTS 函数
        if tts_model == 'bytedance':
            bytedance_tts(text, output_path, speaker_wav)
        elif tts_model == 'aliyun':
            # 使用 aliyun_voice 参数（如果提供）
            aliyun_tts(text, output_path, speaker_wav, voice=aliyun_voice)
        elif tts_model.startswith('xtts_'):
            # XTTS 模型：xtts_v1, xtts_v2, xtts_v2.0.2
            if tts_model == 'xtts_v1':
                model_name = 'tts_models/multilingual/multi-dataset/xtts_v1'
            elif tts_model == 'xtts_v2.0.2':
                model_name = 'tts_models/multilingual/multi-dataset/xtts_v2.0.2'
            else:  # xtts_v2 (默认)
                model_name = xtts_model_name
            xtts_tts(text, output_path, speaker_wav, model_name=model_name)
        else:
            # 默认使用 XTTS v2
            logger.warning(f'未知的 TTS 模型: {tts_model}，使用默认 XTTS v2')
            xtts_tts(text, output_path, speaker_wav, model_name=xtts_model_name)

        start = line['start']
        end = line['end']
        length = end-start
        last_end = len(full_wav)/24000
        if start > last_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000), ))))
        start = len(full_wav)/24000
        line['start'] = start
        if i < len(transcript) - 1:
            next_line = transcript[i+1]
            next_end = next_line['end']
            end = min(start + length, next_end)
        wav, length = adjust_audio_length(output_path, end-start)

        full_wav = np.concatenate((full_wav, wav))
        line['end'] = start + length

    vocal_wav, sr = librosa.load(os.path.join(folder, 'audio_vocals.wav'), sr=24000)
    full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav))
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'))
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    instruments_wav, sr = librosa.load(os.path.join(folder, 'audio_instruments.wav'), sr=24000)
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)

    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(
            instruments_wav, (0, len_full_wav - len_instruments_wav), mode='constant')
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(
            full_wav, (0, len_instruments_wav - len_full_wav), mode='constant')
    combined_wav = full_wav + instruments_wav
    # combined_wav /= np.max(np.abs(combined_wav))
    save_wav_norm(combined_wav, os.path.join(folder, 'audio_combined.wav'))
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')


def generate_all_wavs_under_folder(
    root_folder: str, 
    tts_model: str = 'xtts_v2',
    xtts_model_name: str = 'tts_models/multilingual/multi-dataset/xtts_v2',
    aliyun_voice: str = None
) -> str:
    """为指定文件夹下的所有翻译生成语音
    
    Args:
        root_folder: 根文件夹路径
        tts_model: TTS 模型选择
        xtts_model_name: XTTS 模型路径（仅当 tts_model 为 xtts_* 时有效）
        aliyun_voice: 阿里云 TTS 音色名称（仅当 tts_model 为 aliyun 时有效）
        
    Returns:
        处理结果描述
    """
    logger.info(f'Starting TTS generation for all translations under: {root_folder} (model: {tts_model}, voice: {aliyun_voice})')
    generated_count = 0
    
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files and 'audio_combined.wav' not in files:
            try:
                generate_wavs(root, tts_model=tts_model, xtts_model_name=xtts_model_name, aliyun_voice=aliyun_voice)
                generated_count += 1
            except Exception as e:
                logger.error(f'Failed to generate wavs in {root}: {e}')
    
    result_msg = f'Generated wavs for {generated_count} videos under {root_folder} (using {tts_model})'
    if aliyun_voice:
        result_msg += f' with voice {aliyun_voice}'
    logger.info(result_msg)
    return result_msg


if __name__ == '__main__':
    test_folder = r'videos\TED-Ed\20211214 Would you raise the bird that murdered your children？ - Steve Rothstein'
    generate_wavs(test_folder, tts_model='xtts_v2')
