"""WhisperX 语音识别模块

该模块提供基于 WhisperX 的语音识别功能，包括：
- 语音转文字
- 说话人分离（Diarization）
- 时间对齐
"""

import json
import os
import time
from typing import Optional, Dict, List, Any

import librosa
import numpy as np
import torch
import whisperx
from dotenv import load_dotenv
from loguru import logger

from .utils import save_wav

load_dotenv()

# 全局模型缓存
whisper_model: Optional[Any] = None
diarize_model: Optional[Any] = None
align_model: Optional[Any] = None
language_code: Optional[str] = None
align_metadata: Optional[Dict] = None


def init_whisperx() -> None:
    """初始化所有 WhisperX 相关模型"""
    load_whisper_model()
    load_align_model()
    load_diarize_model()


def load_whisper_model(
    model_name: str = "large-v3", download_root: str = "models/ASR/whisper", device: str = "auto"
) -> None:
    """加载 WhisperX 语音识别模型

    Args:
        model_name: 模型名称，支持 tiny/base/small/medium/large/large-v3
        download_root: 模型下载路径
        device: 设备选择，'auto'/'cuda'/'cpu'
    """
    global whisper_model

    if whisper_model is not None:
        logger.info("WhisperX model already loaded")
        return

    # 规范化模型名称
    if model_name == "large":
        model_name = "large-v3"

    # 自动选择设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading WhisperX model: {model_name} on {device}")
    t_start = time.time()

    whisper_model = whisperx.load_model(model_name, download_root=download_root, device=device, compute_type="int8")

    t_end = time.time()
    logger.info(f"Loaded WhisperX model: {model_name} in {t_end - t_start:.2f}s")


def load_align_model(language: str = "en", device: str = "auto") -> None:
    """加载对齐模型

    Args:
        language: 语言代码
        device: 设备选择，'auto'/'cuda'/'cpu'
    """
    global align_model, language_code, align_metadata

    if align_model is not None and language_code == language:
        logger.info(f"Alignment model for {language} already loaded")
        return

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    language_code = language
    logger.info(f"Loading alignment model for language: {language_code}")
    t_start = time.time()

    align_model, align_metadata = whisperx.load_align_model(language_code=language_code, device=device)

    t_end = time.time()
    logger.info(f"Loaded alignment model: {language_code} in {t_end - t_start:.2f}s")


def load_diarize_model(device: str = "auto") -> None:
    """加载说话人分离模型

    Args:
        device: 设备选择，'auto'/'cuda'/'cpu'
    """
    global diarize_model

    if diarize_model is not None:
        logger.info("Diarization model already loaded")
        return

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading diarization model")
    t_start = time.time()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not found in environment variables")

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)

    t_end = time.time()
    logger.info(f"Loaded diarization model in {t_end - t_start:.2f}s")


def merge_segments(transcript: List[Dict[str, Any]], ending: str = "!\"').:;?]}~") -> List[Dict[str, Any]]:
    """合并语音识别片段，根据标点符号分句

    Args:
        transcript: 原始转录片段列表
        ending: 句子结束标点符号集合

    Returns:
        合并后的转录片段列表
    """
    merged_transcription = []
    buffer_segment = None

    for segment in transcript:
        if buffer_segment is None:
            buffer_segment = segment.copy()
        else:
            # 检查缓冲区最后一个字符是否是标点符号
            if buffer_segment["text"] and buffer_segment["text"][-1] in ending:
                # 是标点符号，保存当前缓冲区并开始新片段
                merged_transcription.append(buffer_segment)
                buffer_segment = segment.copy()
            else:
                # 不是标点符号，继续合并
                buffer_segment["text"] += " " + segment["text"]
                buffer_segment["end"] = segment["end"]

    # 添加最后一个缓冲区
    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription


def transcribe_audio(
    folder: str,
    model_name: str = "large",
    download_root: str = "models/ASR/whisper",
    device: str = "auto",
    batch_size: int = 32,
    diarization: bool = True,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> bool:
    """转录音频文件

    Args:
        folder: 音频文件所在文件夹
        model_name: Whisper 模型名称
        download_root: 模型下载路径
        device: 设备选择
        batch_size: 批处理大小
        diarization: 是否启用说话人分离
        min_speakers: 最小说话人数量
        max_speakers: 最大说话人数量

    Returns:
        是否转录成功
    """
    transcript_path = os.path.join(folder, "transcript.json")
    if os.path.exists(transcript_path):
        logger.info(f"Transcript already exists in {folder}")
        return True

    wav_path = os.path.join(folder, "audio_vocals.wav")
    if not os.path.exists(wav_path):
        logger.warning(f"Audio file not found: {wav_path}")
        return False

    logger.info(f"Transcribing {wav_path}")

    # 自动选择设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型并转录
    load_whisper_model(model_name, download_root, device)
    rec_result = whisper_model.transcribe(wav_path, batch_size=batch_size)

    # 检测语言
    detected_language = rec_result.get("language", "nn")
    if detected_language == "nn":
        logger.warning(f"No language detected in {wav_path}")
        return False

    logger.info(f"Detected language: {detected_language}")

    # 对齐时间戳
    load_align_model(detected_language, device)
    rec_result = whisperx.align(
        rec_result["segments"], align_model, align_metadata, wav_path, device, return_char_alignments=False
    )

    # 说话人分离
    if diarization:
        logger.info("Performing speaker diarization")
        load_diarize_model(device)
        diarize_segments = diarize_model(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
        rec_result = whisperx.assign_word_speakers(diarize_segments, rec_result)

    # 格式化转录结果
    transcript = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "speaker": segment.get("speaker", "SPEAKER_00"),
        }
        for segment in rec_result["segments"]
    ]

    transcript = merge_segments(transcript)

    # 保存转录结果
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)

    logger.info(f"Transcription saved to {transcript_path}")

    # 生成说话人音频
    generate_speaker_audio(folder, transcript)

    return True


def generate_speaker_audio(folder: str, transcript: List[Dict[str, Any]]) -> None:
    """根据转录结果生成每个说话人的音频样本

    Args:
        folder: 工作文件夹路径
        transcript: 转录结果列表
    """
    wav_path = os.path.join(folder, "audio_vocals.wav")

    if not os.path.exists(wav_path):
        logger.warning(f"Vocal audio file not found: {wav_path}")
        return

    logger.info("Generating speaker audio samples")

    # 加载音频数据
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict: Dict[str, np.ndarray] = {}
    audio_length = len(audio_data)
    delay = 0.05  # 在片段前后添加50ms的缓冲

    # 为每个说话人提取音频片段
    for segment in transcript:
        speaker = segment["speaker"]
        start_sample = max(0, int((segment["start"] - delay) * samplerate))
        end_sample = min(int((segment["end"] + delay) * samplerate), audio_length)

        speaker_segment_audio = audio_data[start_sample:end_sample]

        # 拼接同一说话人的所有片段
        if speaker in speaker_dict:
            speaker_dict[speaker] = np.concatenate([speaker_dict[speaker], speaker_segment_audio])
        else:
            speaker_dict[speaker] = speaker_segment_audio

    # 创建说话人文件夹
    speaker_folder = os.path.join(folder, "SPEAKER")
    os.makedirs(speaker_folder, exist_ok=True)

    # 保存每个说话人的音频
    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path)
        logger.info(f"Saved speaker audio: {speaker_file_path}")


def transcribe_all_audio_under_folder(
    folder: str,
    model_name: str = "large",
    download_root: str = "models/ASR/whisper",
    device: str = "auto",
    batch_size: int = 32,
    diarization: bool = True,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> str:
    """转录指定文件夹下所有音频文件

    Args:
        folder: 根文件夹路径
        model_name: Whisper 模型名称
        download_root: 模型下载路径
        device: 设备选择
        batch_size: 批处理大小
        diarization: 是否启用说话人分离
        min_speakers: 最小说话人数量
        max_speakers: 最大说话人数量

    Returns:
        处理结果描述
    """
    logger.info(f"Starting transcription for all audio under: {folder}")
    transcribed_count = 0

    for root, dirs, files in os.walk(folder):
        if "audio_vocals.wav" in files and "transcript.json" not in files:
            success = transcribe_audio(
                root, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers
            )
            if success:
                transcribed_count += 1

    result_msg = f"Transcribed {transcribed_count} audio files under {folder}"
    logger.info(result_msg)
    return result_msg


if __name__ == "__main__":
    transcribe_all_audio_under_folder("videos")
