"""Bilibili 上传模块

该模块提供视频上传到 Bilibili 的功能
"""

import json
import os
import time
from typing import Dict, Any, Optional

from bilibili_toolman.bilisession.web import BiliSession
from bilibili_toolman.bilisession.common.submission import Submission
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def bili_login() -> BiliSession:
    """登录 Bilibili
    
    Returns:
        BiliSession 对象
        
    Raises:
        Exception: 登录失败时抛出异常
    """
    sessdata = os.getenv('BILI_SESSDATA')
    bili_jct = os.getenv('BILI_BILI_JCT')
    
    if not sessdata or not bili_jct:
        raise Exception('BILI_SESSDATA 和 BILI_BILI_JCT 环境变量未设置')
    
    try:
        session = BiliSession(f'SESSDATA={sessdata};bili_jct={bili_jct}')
        logger.info("Bilibili 登录成功")
        return session
    except Exception as e:
        logger.error(f'Bilibili 登录失败: {e}')
        raise Exception('Bilibili 登录失败，请检查 SESSDATA 和 bili_jct 是否正确')

def upload_video(folder: str) -> bool:
    """上传视频到 Bilibili
    
    Args:
        folder: 视频文件所在文件夹
        
    Returns:
        是否上传成功
        
    Raises:
        Exception: 上传失败时抛出异常
    """
    submission_result_path = os.path.join(folder, 'bilibili.json')
    
    # 检查是否已经上传
    if os.path.exists(submission_result_path):
        with open(submission_result_path, 'r', encoding='utf-8') as f:
            submission_result = json.load(f)
        if submission_result.get('results', [{}])[0].get('code') == 0:
            logger.info(f'Video already uploaded: {folder}')
            return True

    # 检查必需的文件
    video_path = os.path.join(folder, 'video.mp4')
    cover_path = os.path.join(folder, 'video.png')
    summary_path = os.path.join(folder, 'summary.json')
    info_path = os.path.join(folder, 'download.info.json')
    
    for path in [video_path, cover_path, summary_path, info_path]:
        if not os.path.exists(path):
            logger.error(f'Required file not found: {path}')
            return False

    # 加载摘要数据
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # 清理标题和摘要
    video_title = summary.get('title', '').replace('视频标题：', '').strip()
    video_summary = summary.get('summary', '').replace(
        '视频摘要：', '').replace('视频简介：', '').strip()
    author = summary.get('author', 'Unknown')
    tags = summary.get('tags', [])
    
    if not isinstance(tags, list):
        tags = []
    
    # 构建标题
    title = f'【中配】{video_title} - {author}'
    
    # 加载原视频信息
    with open(info_path, 'r', encoding='utf-8') as f:
        video_info = json.load(f)
    
    title_english = video_info.get('title', '')
    webpage_url = video_info.get('webpage_url', '')
    
    # 构建描述
    description = (
        f'{title_english}\n'
        f'{video_summary}\n\n'
        f'项目地址：https://github.com/liuzhao1225/YouDub-webui\n'
        f'YouDub 是一个开创性的开源工具，旨在将 YouTube 和其他平台上的高质量视频'
        f'翻译和配音成中文版本。该工具结合了最新的 AI 技术，包括语音识别、'
        f'大型语言模型翻译，以及 AI 声音克隆技术，提供与原视频相似的中文配音，'
        f'为中文用户提供卓越的观看体验。'
    )

    # 登录 Bilibili
    session = bili_login()

    # 提交视频（最多重试 5 次）
    max_retries = 5
    for retry in range(max_retries):
        try:
            logger.info(f'Uploading video (attempt {retry + 1}/{max_retries})')
            
            # 上传视频文件
            video_endpoint, _ = session.UploadVideo(video_path)

            # 创建投稿对象
            submission = Submission(title=title, desc=description)

            # 添加视频
            submission.videos.append(
                Submission(title=title, video_endpoint=video_endpoint)
            )

            # 上传并设置封面
            submission.cover_url = session.UploadCover(cover_path)

            # 设置标签
            all_tags = ['YouDub', author, 'AI', 'ChatGPT'] + tags + ['中文配音', '科学', '科普']
            for tag in all_tags[:12]:  # Bilibili 限制最多 12 个标签
                # 标签长度限制为 20 个字符
                tag = tag[:20] if len(tag) > 20 else tag
                submission.tags.append(tag)
            
            # 设置分区（科普类）
            submission.thread = 201
            
            # 设置版权（转载）
            submission.copyright = submission.COPYRIGHT_REUPLOAD
            submission.source = webpage_url
            
            # 提交投稿
            response = session.SubmitSubmission(submission, seperate_parts=False)
            
            # 检查返回结果
            if response.get('results', [{}])[0].get('code') != 0:
                logger.error(f'Submission failed: {response}')
                raise Exception(f'Submission returned error code: {response}')
            
            logger.info(f"Submission successful: {response}")
            
            # 保存结果
            with open(submission_result_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, ensure_ascii=False, indent=4)
            
            return True
            
        except Exception as e:
            logger.error(f"Upload error (attempt {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                time.sleep(10)
    
    raise Exception(f'上传失败，已重试 {max_retries} 次')


def upload_all_videos_under_folder(folder: str) -> str:
    """上传指定文件夹下的所有视频
    
    Args:
        folder: 根文件夹路径
        
    Returns:
        处理结果描述
    """
    logger.info(f'Starting upload for all videos under: {folder}')
    uploaded_count = 0
    failed_count = 0
    
    for root, _, files in os.walk(folder):
        if 'video.mp4' in files:
            try:
                upload_video(root)
                uploaded_count += 1
            except Exception as e:
                logger.error(f'Failed to upload video in {root}: {e}')
                failed_count += 1
    
    result_msg = f'Upload complete. Success: {uploaded_count}, Failed: {failed_count}'
    logger.info(result_msg)
    return result_msg


if __name__ == '__main__':
    # 示例用法
    test_folder = r'videos\The Game Theorists\20210522 Game Theory What Level is Ashs Pikachu Pokemon'
    upload_all_videos_under_folder(test_folder)
