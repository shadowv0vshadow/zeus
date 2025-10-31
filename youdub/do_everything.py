"""全流程处理模块

该模块提供视频处理的完整流程，包括：
- 视频下载
- 音频分离
- 语音识别
- 翻译
- 语音合成
- 视频合成
- 上传到 Bilibili
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

from loguru import logger

from .step000_video_downloader import (
    get_info_list_from_url,
    download_single_video,
    get_target_folder
)
from .step010_demucs_vr import separate_all_audio_under_folder, init_demucs
from .step020_whisperx import transcribe_all_audio_under_folder, init_whisperx
from .step030_translation import translate_all_transcript_under_folder
from .step040_tts import generate_all_wavs_under_folder
from .step042_tts_xtts import init_TTS
from .step050_synthesize_video import synthesize_all_video_under_folder
from .step060_genrate_info import generate_all_info_under_folder
from .step070_upload_bilibili import upload_all_videos_under_folder



def process_video(
    info: Dict[str, Any],
    root_folder: str,
    resolution: str,
    demucs_model: str,
    device: str,
    shifts: int,
    whisper_model: str,
    whisper_download_root: str,
    whisper_batch_size: int,
    whisper_diarization: bool,
    whisper_min_speakers: Optional[int],
    whisper_max_speakers: Optional[int],
    translation_target_language: str,
    translation_model_provider: str,
    tts_model: str,
    xtts_model_name: str,
    aliyun_voice: Optional[str] = None,
    subtitles: bool = True,
    speed_up: float = 1.05,
    fps: int = 30,
    target_resolution: str = '1080p',
    max_retries: int = 5,
    auto_upload_video: bool = True
) -> bool:
    """处理单个视频的完整流程
    
    Args:
        info: 视频信息字典
        root_folder: 根文件夹路径
        resolution: 下载分辨率
        demucs_model: Demucs 模型名称
        device: 设备选择
        shifts: Demucs shifts 参数
        whisper_model: Whisper 模型名称
        whisper_download_root: Whisper 模型下载路径
        whisper_batch_size: Whisper 批处理大小
        whisper_diarization: 是否启用说话人分离
        whisper_min_speakers: 最小说话人数量
        whisper_max_speakers: 最大说话人数量
        translation_target_language: 目标翻译语言
        tts_model: TTS 模型选择
        xtts_model_name: XTTS 模型路径
        subtitles: 是否添加字幕
        speed_up: 视频加速倍率
        fps: 目标帧率
        target_resolution: 输出分辨率
        max_retries: 最大重试次数
        auto_upload_video: 是否自动上传视频
        
    Returns:
        是否处理成功
    """
    for retry in range(max_retries):
        try:
            # 获取目标文件夹
            folder = get_target_folder(info, root_folder)
            if folder is None:
                logger.warning(f'Failed to get target folder for video {info["title"]}')
                return False

            # 检查是否已经上传
            bilibili_json_path = os.path.join(folder, 'bilibili.json')
            if os.path.exists(bilibili_json_path):
                with open(bilibili_json_path, 'r', encoding='utf-8') as f:
                    bilibili_info = json.load(f)
                if bilibili_info.get('results', [{}])[0].get('code') == 0:
                    logger.info(f'Video already uploaded in {folder}')
                    return True

            # 下载视频
            folder = download_single_video(info, root_folder, resolution)
            if folder is None:
                logger.warning(f'Failed to download video {info["title"]}')
                return True

            logger.info(f'Processing video in {folder}')
            
            # 1. 音频分离
            separate_all_audio_under_folder(
                folder,
                model_name=demucs_model,
                device=device,
                progress=True,
                shifts=shifts
            )
            
            # 2. 语音识别
            transcribe_all_audio_under_folder(
                folder,
                model_name=whisper_model,
                download_root=whisper_download_root,
                device=device,
                batch_size=whisper_batch_size,
                diarization=whisper_diarization,
                min_speakers=whisper_min_speakers,
                max_speakers=whisper_max_speakers
            )

            # 3. 翻译
            translate_all_transcript_under_folder(
                folder,
                target_language=translation_target_language,
                model_provider=translation_model_provider
            )
            
            # 4. 语音合成
            generate_all_wavs_under_folder(
                folder,
                tts_model=tts_model,
                xtts_model_name=xtts_model_name,
                aliyun_voice=aliyun_voice
            )
            
            # 5. 视频合成
            synthesize_all_video_under_folder(
                folder,
                subtitles=subtitles,
                speed_up=speed_up,
                fps=fps,
                resolution=target_resolution
            )
            
            # 6. 生成信息
            generate_all_info_under_folder(folder)
            
            # 7. 上传到 Bilibili
            if auto_upload_video:
                time.sleep(1)
                upload_all_videos_under_folder(folder)
                
            return True
            
        except Exception as e:
            logger.error(f'Error processing video {info["title"]} (retry {retry + 1}/{max_retries}): {e}')
            if retry < max_retries - 1:
                time.sleep(5)
    
    return False


def do_everything(
    root_folder: str,
    url: str,
    num_videos: int = 5,
    resolution: str = '1080p',
    demucs_model: str = 'htdemucs_ft',
    device: str = 'auto',
    shifts: int = 5,
    whisper_model: str = 'large',
    whisper_download_root: str = 'models/ASR/whisper',
    whisper_batch_size: int = 32,
    whisper_diarization: bool = True,
    whisper_min_speakers: Optional[int] = None,
    whisper_max_speakers: Optional[int] = None,
    translation_target_language: str = '简体中文',
    translation_model_provider: str = 'openai',
    tts_model: str = 'xtts_v2',
    xtts_model_name: str = 'tts_models/multilingual/multi-dataset/xtts_v2',
    aliyun_voice: Optional[str] = None,
    subtitles: bool = True,
    speed_up: float = 1.05,
    fps: int = 30,
    target_resolution: str = '1080p',
    max_workers: int = 3,
    max_retries: int = 5,
    auto_upload_video: bool = True
) -> str:
    """执行完整的视频处理流程
    
    Args:
        root_folder: 根文件夹路径
        url: 视频 URL（支持多个，用逗号分隔）
        num_videos: 要下载的视频数量
        resolution: 下载分辨率
        demucs_model: Demucs 模型名称
        device: 设备选择
        shifts: Demucs shifts 参数
        whisper_model: Whisper 模型名称
        whisper_download_root: Whisper 模型下载路径
        whisper_batch_size: Whisper 批处理大小
        whisper_diarization: 是否启用说话人分离
        whisper_min_speakers: 最小说话人数量
        whisper_max_speakers: 最大说话人数量
        translation_target_language: 目标翻译语言
        tts_model: TTS 模型选择
        xtts_model_name: XTTS 模型路径
        subtitles: 是否添加字幕
        speed_up: 视频加速倍率
        fps: 目标帧率
        target_resolution: 输出分辨率
        max_workers: 最大并行任务数
        max_retries: 最大重试次数
        auto_upload_video: 是否自动上传视频
        
    Returns:
        处理结果摘要
    """
    success_list = []
    fail_list = []

    # 解析 URL 列表
    url = url.replace(' ', '').replace('，', '\n').replace(',', '\n')
    urls = [u for u in url.split('\n') if u.strip()]

    logger.info(f'Starting processing for {len(urls)} URL(s)')
    
    # 使用线程池预加载模型
    with ThreadPoolExecutor(max_workers=3) as executor:
        logger.info('Pre-loading models in parallel')
        executor.submit(init_demucs)
        executor.submit(init_TTS)
        executor.submit(init_whisperx)

    # 处理每个视频
    for info in get_info_list_from_url(urls, num_videos):
        success = process_video(
            info,
            root_folder,
            resolution,
            demucs_model,
            device,
            shifts,
            whisper_model,
            whisper_download_root,
            whisper_batch_size,
            whisper_diarization,
            whisper_min_speakers,
            whisper_max_speakers,
            translation_target_language,
            translation_model_provider,
            tts_model,
            xtts_model_name,
            aliyun_voice,
            subtitles,
            speed_up,
            fps,
            target_resolution,
            max_retries,
            auto_upload_video
        )
        
        if success:
            success_list.append(info)
        else:
            fail_list.append(info)

    result_msg = f'Success: {len(success_list)}\nFail: {len(fail_list)}'
    logger.info(result_msg)
    return result_msg
