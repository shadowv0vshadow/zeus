"""TTS（文本转语音）模块

该模块提供文本转语音功能，支持：
- 字节跳动 TTS
- XTTS 模型
- 音频时长调整
"""

import json
import os
import re
import glob
from typing import List, Dict, Any, Optional

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
    # 如果文本为空，直接返回
    if not text or not text.strip():
        return text
    
    # 替换特殊词汇
    text = text.replace("AI", "人工智能")

    # 在大写字母前插入空格（除了开头）
    text = re.sub(r"(?<!^)([A-Z])", r" \1", text)

    # 文本规范化
    text = normalizer(text)

    # 在字母和数字之间插入空格
    text = re.sub(r"(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])", " ", text)

    return text


def adjust_audio_length(
    wav_path: str,
    desired_length: float,
    sample_rate: int = 24000,
    min_speed_factor: float = 0.6,
    max_speed_factor: float = 1.1,
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
    target_path = wav_path.replace(".wav", "_adjusted.wav")

    # 拉伸音频
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)

    # 重新加载调整后的音频
    wav, sample_rate = librosa.load(target_path, sr=sample_rate)

    # 截取到目标长度
    target_samples = int(actual_length * sample_rate)
    return wav[:target_samples], actual_length


def generate_wavs(
    folder: str,
    tts_model: str = "xtts_v2",
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    aliyun_voice: str = None,
) -> None:
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
    transcript_path = os.path.join(folder, "translation.json")
    output_folder = os.path.join(folder, "wavs")
    os.makedirs(output_folder, exist_ok=True)
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    speakers = set()

    for line in transcript:
        speakers.add(line["speaker"])
    num_speakers = len(speakers)
    logger.info(f"Found {num_speakers} speakers")

    # 加载原始人声音频（用于提取原始片段）
    vocal_wav_path = os.path.join(folder, "audio_vocals.wav")
    original_vocal_audio = None
    vocal_sample_rate = 24000
    if os.path.exists(vocal_wav_path):
        try:
            original_vocal_audio, vocal_sample_rate = librosa.load(vocal_wav_path, sr=24000)
            logger.info(f"Loaded original vocal audio: {len(original_vocal_audio)/vocal_sample_rate:.2f}s")
        except Exception as e:
            logger.warning(f"Failed to load original vocal audio: {e}")

    full_wav = np.zeros((0,))
    for i, line in enumerate(transcript):
        speaker = line["speaker"]
        translation_text = line.get("translation", "").strip()
        
        # 如果翻译为空，使用原文
        if not translation_text:
            logger.warning(f"片段 {i:04d} 的翻译为空，使用原文: {line.get('text', '')[:50]}...")
            translation_text = line.get("text", "")
        
        # 如果原文也为空，创建一个静音文件并继续
        if not translation_text.strip():
            logger.warning(f"片段 {i:04d} 的原文和翻译都为空，创建静音文件")
            output_path = os.path.join(output_folder, f"{str(i).zfill(4)}.wav")
            # 计算片段时长
            start = line["start"]
            end = line["end"]
            duration = max(0.1, end - start)  # 至少0.1秒
            silence_samples = int(duration * 24000)
            save_wav(np.zeros(silence_samples), output_path, sample_rate=24000)
            
            # 更新时间信息
            last_end = len(full_wav) / 24000
            if start > last_end:
                full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000),))))
            start = len(full_wav) / 24000
            line["start"] = start
            wav = np.zeros(silence_samples)
            full_wav = np.concatenate((full_wav, wav))
            line["end"] = start + duration
            continue
        
        text = preprocess_text(translation_text)
        output_path = os.path.join(output_folder, f"{str(i).zfill(4)}.wav")
        speaker_wav = os.path.join(folder, "SPEAKER", f"{speaker}.wav")

        # 提取并保存原始音频片段（如果原始音频可用）
        if original_vocal_audio is not None:
            original_output_path = os.path.join(output_folder, f"{str(i).zfill(4)}_original.wav")
            try:
                start_time = line["start"]
                end_time = line["end"]
                start_sample = int(start_time * vocal_sample_rate)
                end_sample = int(end_time * vocal_sample_rate)

                # 确保索引在有效范围内
                start_sample = max(0, min(start_sample, len(original_vocal_audio)))
                end_sample = max(start_sample, min(end_sample, len(original_vocal_audio)))

                if end_sample > start_sample:
                    original_segment = original_vocal_audio[start_sample:end_sample]
                    save_wav(original_segment, original_output_path, sample_rate=vocal_sample_rate)
                    logger.debug(
                        f"Saved original segment {i}: {original_output_path} ({len(original_segment)/vocal_sample_rate:.2f}s)"
                    )
            except Exception as e:
                logger.warning(f"Failed to extract original segment {i}: {e}")

        # 根据选择的 TTS 模型调用对应的 TTS 函数
        if tts_model == "bytedance":
            bytedance_tts(text, output_path, speaker_wav)
        elif tts_model == "aliyun":
            # 使用 aliyun_voice 参数（如果提供）
            aliyun_tts(text, output_path, speaker_wav, voice=aliyun_voice)
        elif tts_model.startswith("xtts_"):
            # XTTS 模型：xtts_v1, xtts_v2, xtts_v2.0.2
            if tts_model == "xtts_v1":
                model_name = "tts_models/multilingual/multi-dataset/xtts_v1"
            elif tts_model == "xtts_v2.0.2":
                model_name = "tts_models/multilingual/multi-dataset/xtts_v2.0.2"
            else:  # xtts_v2 (默认)
                model_name = xtts_model_name
            xtts_tts(text, output_path, speaker_wav, model_name=model_name)
        else:
            # 默认使用 XTTS v2
            logger.warning(f"未知的 TTS 模型: {tts_model}，使用默认 XTTS v2")
            xtts_tts(text, output_path, speaker_wav, model_name=xtts_model_name)

        start = line["start"]
        end = line["end"]
        length = end - start
        last_end = len(full_wav) / 24000
        if start > last_end:
            full_wav = np.concatenate((full_wav, np.zeros((int((start - last_end) * 24000),))))
        start = len(full_wav) / 24000
        line["start"] = start
        if i < len(transcript) - 1:
            next_line = transcript[i + 1]
            next_end = next_line["end"]
            end = min(start + length, next_end)
        wav, length = adjust_audio_length(output_path, end - start)

        full_wav = np.concatenate((full_wav, wav))
        line["end"] = start + length

    vocal_wav, sr = librosa.load(os.path.join(folder, "audio_vocals.wav"), sr=24000)
    full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav))
    save_wav(full_wav, os.path.join(folder, "audio_tts.wav"))
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    instruments_wav, sr = librosa.load(os.path.join(folder, "audio_instruments.wav"), sr=24000)
    len_full_wav = len(full_wav)
    len_instruments_wav = len(instruments_wav)

    if len_full_wav > len_instruments_wav:
        # 如果 full_wav 更长，将 instruments_wav 延伸到相同长度
        instruments_wav = np.pad(instruments_wav, (0, len_full_wav - len_instruments_wav), mode="constant")
    elif len_instruments_wav > len_full_wav:
        # 如果 instruments_wav 更长，将 full_wav 延伸到相同长度
        full_wav = np.pad(full_wav, (0, len_instruments_wav - len_full_wav), mode="constant")
    combined_wav = full_wav + instruments_wav
    # combined_wav /= np.max(np.abs(combined_wav))
    save_wav_norm(combined_wav, os.path.join(folder, "audio_combined.wav"))
    logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')


def get_all_projects(root_folder: str = "videos") -> List[Dict[str, Any]]:
    """获取所有项目列表

    Args:
        root_folder: 根文件夹路径

    Returns:
        项目列表，每个项目包含路径和基本信息
    """
    projects = []

    if not os.path.exists(root_folder):
        return projects

    for root, dirs, files in os.walk(root_folder):
        # 检查是否是项目文件夹（有 translation.json 和 wavs 文件夹）
        translation_path = os.path.join(root, "translation.json")
        wavs_folder = os.path.join(root, "wavs")

        if os.path.exists(translation_path) and os.path.exists(wavs_folder):
            # 获取项目名称（相对路径）
            project_name = os.path.relpath(root, root_folder)

            # 获取项目标题（从 download.info.json）
            title = project_name
            info_path = os.path.join(root, "download.info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                        title = info.get("title", project_name)
                except:
                    pass

            projects.append({"path": root, "name": project_name, "title": title, "relative_path": project_name})

    return sorted(projects, key=lambda x: x["name"])


def get_audio_segments(project_folder: str) -> List[Dict[str, Any]]:
    """从translation.json获取所有音频片段信息

    Args:
        project_folder: 项目文件夹路径

    Returns:
        音频片段列表，包含编号、翻译文本、文件路径等信息
    """
    wavs_folder = os.path.join(project_folder, "wavs")
    translation_path = os.path.join(project_folder, "translation.json")

    if not os.path.exists(translation_path):
        return []

    segments = []

    # 从translation.json读取所有片段
    try:
        with open(translation_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    except:
        return []

    # 遍历translation.json中的每个片段
    for i, line in enumerate(transcript):
        segment_id = f"{i:04d}"  # 格式化为0000, 0001等

        # 检查AI语音文件（0000.wav - 已调整好的）
        ai_voice_file = os.path.join(wavs_folder, f"{segment_id}.wav")
        has_ai_voice = os.path.exists(ai_voice_file)

        # 检查原声文件（0000_original.wav）
        original_file = os.path.join(wavs_folder, f"{segment_id}_original.wav")
        has_original = os.path.exists(original_file)

        # 获取翻译文本
        translation_text = line.get("translation", "")

        segments.append(
            {
                "id": segment_id,
                "number": i,
                "ai_voice_path": ai_voice_file if has_ai_voice else None,
                "original_voice_path": original_file if has_original else None,
                "has_ai_voice": has_ai_voice,
                "has_original": has_original,
                "translation": translation_text[:50] + "..." if len(translation_text) > 50 else translation_text,
            }
        )

    return segments


def load_audio_config(project_folder: str) -> Dict[str, bool]:
    """从translation.json加载音频配置（优先），如果不存在则从audio_config.json加载（向后兼容）

    Args:
        project_folder: 项目文件夹路径

    Returns:
        音频配置字典，键为segment_id（如"0000"），值为是否使用原声（bool）
    """
    # 优先从translation.json读取use_original字段
    translation_path = os.path.join(project_folder, "translation.json")
    config = {}

    if os.path.exists(translation_path):
        try:
            with open(translation_path, "r", encoding="utf-8") as f:
                translation_data = json.load(f)

            for i, line in enumerate(translation_data):
                segment_id = f"{i:04d}"
                use_original = line.get("use_original", False)
                config[segment_id] = use_original

            if config:
                logger.debug(f"Loaded audio config from translation.json: {len(config)} segments")
                return config
        except Exception as e:
            logger.warning(f"Failed to load audio config from translation.json: {e}")

    # 如果没有从translation.json加载到配置，尝试从audio_config.json加载（向后兼容）
    config_path = os.path.join(project_folder, "audio_config.json")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.debug(f"Loaded audio config from audio_config.json: {len(config)} segments")
                return config
        except Exception as e:
            logger.warning(f"Failed to load audio config from audio_config.json: {e}")

    # 默认配置：所有片段都使用AI配音
    return {}


def save_audio_config(project_folder: str, config: Dict[str, bool]) -> None:
    """保存音频选择配置

    Args:
        project_folder: 项目文件夹路径
        config: 配置字典，key为片段ID，value为True表示使用原声，False表示使用AI配音
    """
    config_path = os.path.join(project_folder, "audio_config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def regenerate_combined_audio(folder: str, audio_config: Optional[Dict[str, bool]] = None) -> None:
    """根据配置重新生成 audio_combined.wav

    Args:
        folder: 项目文件夹路径
        audio_config: 音频选择配置，如果为None则使用默认（全部AI配音）
    """
    wavs_folder = os.path.join(folder, "wavs")
    translation_path = os.path.join(folder, "translation.json")

    if not os.path.exists(wavs_folder):
        logger.error(f"Wavs folder not found: {wavs_folder}")
        return

    # 加载配置
    if audio_config is None:
        audio_config = load_audio_config(folder)

    # 打印接收到的配置（用于调试）
    logger.info(f"regenerate_combined_audio 接收到的配置: {audio_config}")

    # 加载翻译数据
    with open(translation_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    # 获取所有音频片段
    segments = get_audio_segments(folder)

    if not segments:
        logger.error(f"No audio segments found in {wavs_folder}")
        return

    logger.info(f"Regenerating combined audio for {len(segments)} segments")

    # 根据配置生成最终带编号的wav文件（覆盖原有的0000.wav）
    for segment in segments:
        segment_id = segment["id"]

        # 根据配置选择使用原声还是AI配音
        use_original = audio_config.get(segment_id, False)

        # 尝试多种格式的 segment_id 查找配置
        if not use_original:
            try:
                segment_num = int(segment_id)
                use_original = audio_config.get(str(segment_num), False)
            except:
                pass

        # 确定源文件路径
        source_path = None
        if use_original and segment.get("has_original") and segment.get("original_voice_path"):
            source_path = segment["original_voice_path"]
            logger.debug(f"Segment {segment_id}: 使用原声 -> {source_path}")
        elif segment.get("has_ai_voice") and segment.get("ai_voice_path"):
            source_path = segment["ai_voice_path"]
            logger.debug(f"Segment {segment_id}: 使用AI配音 -> {source_path}")

        # 生成最终的编号文件（覆盖原有的0000.wav）
        if source_path and os.path.exists(source_path):
            final_wav_path = os.path.join(wavs_folder, f"{segment_id}.wav")
            if source_path != final_wav_path:
                try:
                    wav_data, sr = librosa.load(source_path, sr=24000)
                    save_wav(wav_data, final_wav_path, sample_rate=24000)
                    logger.info(f'✅ 生成最终文件: {final_wav_path} ({"原声" if use_original else "AI配音"})')
                except Exception as e:
                    logger.warning(f"Failed to generate final wav file for {segment_id}: {e}")
            else:
                logger.debug(f"Segment {segment_id}: 源文件已是最终文件")

    # 组合音频
    full_wav = np.zeros((0,))
    sample_rate = 24000

    for segment in segments:
        segment_id = segment["id"]
        segment_num = segment["number"]

        # 现在直接使用生成的最终编号文件（0000.wav）
        audio_path = os.path.join(wavs_folder, f"{segment_id}.wav")

        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            continue

        # 获取对应的时间信息（优先从transcript获取）
        if segment_num < len(transcript):
            line = transcript[segment_num]
            start_time = line.get("start", 0)
            end_time = line.get("end", 0)
        else:
            # 如果transcript中没有，使用累积时间
            start_time = len(full_wav) / sample_rate
            try:
                wav_temp, sr_temp = librosa.load(audio_path, sr=sample_rate)
                duration = len(wav_temp) / sr_temp
                end_time = start_time + duration
            except:
                continue

        # 加载音频
        try:
            wav, sr = librosa.load(audio_path, sr=sample_rate)
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")
            continue

        # 添加静音间隔（如果需要）
        last_end = len(full_wav) / sample_rate
        if start_time > last_end:
            silence_samples = int((start_time - last_end) * sample_rate)
            full_wav = np.concatenate((full_wav, np.zeros(silence_samples)))

        # 调整音频长度以匹配目标时长
        current_start = len(full_wav) / sample_rate
        target_length = end_time - current_start
        if target_length > 0:
            current_length = len(wav) / sample_rate
            if abs(current_length - target_length) > 0.1:  # 如果差异超过0.1秒，需要调整
                # 使用audiostretchy调整长度
                from audiostretchy.stretch import stretch_audio

                temp_path = os.path.join(wavs_folder, f"{segment_id}_temp.wav")
                speed_factor = current_length / target_length
                speed_factor = max(0.6, min(1.1, speed_factor))  # 限制范围
                try:
                    stretch_audio(audio_path, temp_path, ratio=speed_factor, sample_rate=sample_rate)
                    wav, sr = librosa.load(temp_path, sr=sample_rate)
                    # 截取到目标长度
                    target_samples = int(target_length * sample_rate)
                    if len(wav) > target_samples:
                        wav = wav[:target_samples]
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    logger.debug(f"Failed to adjust audio length for {segment_id}: {e}")
                    # 如果调整失败，使用原始音频，但截取到目标长度
                    target_samples = int(target_length * sample_rate)
                    if len(wav) > target_samples:
                        wav = wav[:target_samples]

        # 拼接音频
        full_wav = np.concatenate((full_wav, wav))

    # 加载人声音频用于音量归一化
    vocal_wav_path = os.path.join(folder, "audio_vocals.wav")
    if os.path.exists(vocal_wav_path):
        try:
            vocal_wav, sr = librosa.load(vocal_wav_path, sr=24000)
            if len(vocal_wav) > 0:
                full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav))
        except:
            pass

    # 保存TTS音频
    save_wav(full_wav, os.path.join(folder, "audio_tts.wav"))

    # 与伴奏合并
    instruments_wav_path = os.path.join(folder, "audio_instruments.wav")
    if os.path.exists(instruments_wav_path):
        try:
            instruments_wav, sr = librosa.load(instruments_wav_path, sr=24000)
            len_full_wav = len(full_wav)
            len_instruments_wav = len(instruments_wav)

            if len_full_wav > len_instruments_wav:
                instruments_wav = np.pad(instruments_wav, (0, len_full_wav - len_instruments_wav), mode="constant")
            elif len_instruments_wav > len_full_wav:
                full_wav = np.pad(full_wav, (0, len_instruments_wav - len_full_wav), mode="constant")

            combined_wav = full_wav + instruments_wav
            save_wav_norm(combined_wav, os.path.join(folder, "audio_combined.wav"))
            logger.info(f"Regenerated audio_combined.wav for {folder}")
        except Exception as e:
            logger.error(f"Failed to merge with instruments: {e}")
    else:
        logger.warning(f"Instruments audio not found: {instruments_wav_path}")


def generate_all_wavs_under_folder(
    root_folder: str,
    tts_model: str = "xtts_v2",
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
    aliyun_voice: str = None,
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
    logger.info(
        f"Starting TTS generation for all translations under: {root_folder} (model: {tts_model}, voice: {aliyun_voice})"
    )
    generated_count = 0

    for root, dirs, files in os.walk(root_folder):
        if "translation.json" in files and "audio_combined.wav" not in files:
            try:
                generate_wavs(root, tts_model=tts_model, xtts_model_name=xtts_model_name, aliyun_voice=aliyun_voice)
                generated_count += 1
            except Exception as e:
                logger.error(f"Failed to generate wavs in {root}: {e}")

    result_msg = f"Generated wavs for {generated_count} videos under {root_folder} (using {tts_model})"
    if aliyun_voice:
        result_msg += f" with voice {aliyun_voice}"
    logger.info(result_msg)
    return result_msg


if __name__ == "__main__":
    test_folder = r"videos\TED-Ed\20211214 Would you raise the bird that murdered your children？ - Steve Rothstein"
    generate_wavs(test_folder, tts_model="xtts_v2")
