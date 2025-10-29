"""视频下载模块

该模块提供从 YouTube 和其他平台下载视频的功能
"""

import os
import re
from typing import Optional, List, Dict, Any, Generator

import yt_dlp
from loguru import logger


def sanitize_title(title: str) -> str:
    """清理视频标题，只保留合法字符
    
    Args:
        title: 原始标题
        
    Returns:
        清理后的标题
    """
    # 只保留数字、字母、中文字符、空格、下划线和连字符
    title = re.sub(r'[^\w\u4e00-\u9fff \d_-]', '', title)
    # 将多个空格替换为单个空格
    title = re.sub(r'\s+', ' ', title)
    return title.strip()


def get_target_folder(info: Dict[str, Any], folder_path: str) -> Optional[str]:
    """根据视频信息获取目标文件夹路径
    
    Args:
        info: 视频信息字典
        folder_path: 根文件夹路径
        
    Returns:
        目标文件夹路径，如果无法确定则返回 None
    """
    sanitized_title = sanitize_title(info.get('title', 'Unknown'))
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    
    if upload_date == 'Unknown':
        logger.warning(f'No upload date found for video: {sanitized_title}')
        return None

    output_folder = os.path.join(
        folder_path,
        sanitized_uploader,
        f'{upload_date} {sanitized_title}'
    )

    return output_folder


def download_single_video(
    info: Dict[str, Any],
    folder_path: str,
    resolution: str = '1080p'
) -> Optional[str]:
    """下载单个视频
    
    Args:
        info: 视频信息字典
        folder_path: 根文件夹路径
        resolution: 目标分辨率
        
    Returns:
        下载文件夹路径，失败则返回 None
    """
    sanitized_title = sanitize_title(info.get('title', 'Unknown'))
    sanitized_uploader = sanitize_title(info.get('uploader', 'Unknown'))
    upload_date = info.get('upload_date', 'Unknown')
    
    if upload_date == 'Unknown':
        logger.warning(f'No upload date found for video: {sanitized_title}')
        return None

    output_folder = os.path.join(
        folder_path,
        sanitized_uploader,
        f'{upload_date} {sanitized_title}'
    )
    
    # 检查是否已下载
    if os.path.exists(os.path.join(output_folder, 'download.mp4')):
        logger.info(f'Video already downloaded in {output_folder}')
        return output_folder

    # 准备下载选项
    resolution_value = resolution.replace('p', '')
    ydl_opts = {
        'format': (
            f'bestvideo[ext=mp4][height<={resolution_value}]+bestaudio[ext=m4a]/'
            f'best[ext=mp4]/best'
        ),
        'writeinfojson': True,
        'writethumbnail': True,
        'outtmpl': os.path.join(output_folder, 'download'),
        'ignoreerrors': True
    }

    # 下载视频
    logger.info(f'Downloading video: {sanitized_title}')
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([info['webpage_url']])
        logger.info(f'Video downloaded to {output_folder}')
        return output_folder
    except Exception as e:
        logger.error(f'Failed to download video {sanitized_title}: {e}')
        return None


def download_videos(
    info_list: List[Dict[str, Any]],
    folder_path: str,
    resolution: str = '1080p'
) -> None:
    """下载多个视频
    
    Args:
        info_list: 视频信息列表
        folder_path: 根文件夹路径
        resolution: 目标分辨率
    """
    for info in info_list:
        download_single_video(info, folder_path, resolution)


def get_info_list_from_url(
    url: List[str],
    num_videos: int
) -> Generator[Dict[str, Any], None, None]:
    """从 URL 获取视频信息列表
    
    Args:
        url: URL 列表
        num_videos: 要获取的视频数量
        
    Yields:
        视频信息字典
    """
    if isinstance(url, str):
        url = [url]

    # 准备选项
    ydl_opts = {
        'format': 'best',
        'dumpjson': True,
        'playlistend': num_videos,
        'ignoreerrors': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            try:
                result = ydl.extract_info(u, download=False)
                
                if 'entries' in result:
                    # 播放列表
                    for video_info in result['entries']:
                        if video_info:  # 过滤掉 None 值
                            yield video_info
                else:
                    # 单个视频
                    yield result
                    
            except Exception as e:
                logger.error(f'Failed to extract info from {u}: {e}')


def download_from_url(
    url: str,
    folder_path: str,
    resolution: str = '1080p',
    num_videos: int = 5
) -> None:
    """从 URL 下载视频
    
    Args:
        url: 视频或播放列表 URL
        folder_path: 输出文件夹路径
        resolution: 目标分辨率
        num_videos: 要下载的视频数量
    """
    if isinstance(url, str):
        url = [url]

    logger.info(f'Fetching video information from {len(url)} URL(s)')
    
    # 获取视频信息
    ydl_opts = {
        'format': 'best',
        'dumpjson': True,
        'playlistend': num_videos,
        'ignoreerrors': True
    }

    video_info_list = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            try:
                result = ydl.extract_info(u, download=False)
                
                if 'entries' in result:
                    # 播放列表
                    video_info_list.extend([v for v in result['entries'] if v])
                else:
                    # 单个视频
                    video_info_list.append(result)
                    
            except Exception as e:
                logger.error(f'Failed to extract info from {u}: {e}')

    # 下载视频
    logger.info(f'Downloading {len(video_info_list)} video(s)')
    download_videos(video_info_list, folder_path, resolution)


if __name__ == '__main__':
    # 示例用法
    test_url = 'https://www.youtube.com/watch?v=3LPJfIKxwWc'
    test_folder = 'videos'
    download_from_url(test_url, test_folder)
