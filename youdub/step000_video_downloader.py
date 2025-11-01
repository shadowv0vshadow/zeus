"""视频下载模块

该模块提供从 YouTube 和其他平台下载视频的功能
"""

import os
import re
import platform
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

    # 获取 cookies 配置
    cookies_opts = _get_cookies_options()

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
        'ignoreerrors': True,
        **cookies_opts  # 添加 cookies 配置
    }

    # 下载视频
    logger.info(f'Downloading video: {sanitized_title}')
    try:
        webpage_url = info.get('webpage_url') or info.get('url')
        if not webpage_url:
            logger.error(f'No webpage_url found in video info: {sanitized_title}')
            return None
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([webpage_url])
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

    # 获取 cookies 配置
    cookies_opts = _get_cookies_options()

    for u in url:
        try:
            # 检测URL类型：如果包含 watch?v=，视为单个视频；否则视为播放列表
            is_single_video = 'watch?v=' in u or '/watch/' in u
            
            # 准备选项
            ydl_opts = {
                'format': 'best',
                'dumpjson': True,
                'ignoreerrors': True,
                **cookies_opts  # 添加 cookies 配置
            }
            
            if is_single_video:
                # 单个视频：即使URL包含播放列表参数，也只下载该视频
                ydl_opts['noplaylist'] = True
            else:
                # 播放列表：限制下载数量
                ydl_opts['playlistend'] = num_videos
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(u, download=False)
            
            # 检查 result 是否为 None
            if result is None:
                logger.error(f'Failed to extract info from {u}: extract_info returned None')
                continue
            
            if 'entries' in result:
                # 播放列表
                entries = result.get('entries', [])
                for video_info in entries:
                    if video_info:  # 过滤掉 None 值
                        yield video_info
            else:
                # 单个视频
                yield result
                    
        except Exception as e:
            logger.error(f'Failed to extract info from {u}: {e}')


def _get_cookies_options() -> Dict[str, Any]:
    """获取 cookies 配置选项
    
    Returns:
        cookies 相关配置字典
    """
    cookies_opts = {}
    
    # 尝试从浏览器自动提取 cookies（优先 Chrome/Edge）
    try:
        system = platform.system()
        if system == 'Darwin':  # macOS
            # macOS Chrome
            chrome_cookies = os.path.expanduser('~/Library/Application Support/Google/Chrome/Default/Cookies')
            if os.path.exists(chrome_cookies):
                cookies_opts['cookiesfrombrowser'] = ('chrome',)
                logger.info('使用 Chrome cookies (macOS)')
                return cookies_opts
        elif system == 'Linux':
            # Linux Chrome
            chrome_cookies = os.path.expanduser('~/.config/google-chrome/Default/Cookies')
            if os.path.exists(chrome_cookies):
                cookies_opts['cookiesfrombrowser'] = ('chrome',)
                logger.info('使用 Chrome cookies (Linux)')
                return cookies_opts
        elif system == 'Windows':
            # Windows Chrome
            chrome_cookies = os.path.expanduser('~/AppData/Local/Google/Chrome/User Data/Default/Cookies')
            if os.path.exists(chrome_cookies):
                cookies_opts['cookiesfrombrowser'] = ('chrome',)
                logger.info('使用 Chrome cookies (Windows)')
                return cookies_opts
    except Exception as e:
        logger.debug(f'无法自动检测浏览器 cookies: {e}')
    
    # 检查环境变量中的 cookies 文件路径
    cookies_file = os.getenv('YOUTUBE_COOKIES_FILE')
    if cookies_file and os.path.exists(cookies_file):
        cookies_opts['cookies'] = cookies_file
        logger.info(f'使用自定义 cookies 文件: {cookies_file}')
        return cookies_opts
    
    # 如果都没有，返回空字典（不使用 cookies）
    logger.warning('未检测到浏览器 cookies，可能会遇到 YouTube bot 验证。建议：')
    logger.warning('1. 设置环境变量 YOUTUBE_COOKIES_FILE 指向 cookies.txt 文件')
    logger.warning('2. 或使用浏览器扩展导出 cookies（如 Get cookies.txt LOCALLY）')
    return cookies_opts


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
    
    # 获取 cookies 配置
    cookies_opts = _get_cookies_options()
    
    video_info_list = []
    for u in url:
        try:
            # 检测URL类型：如果包含 watch?v=，视为单个视频；否则视为播放列表
            is_single_video = 'watch?v=' in u or '/watch/' in u
            
            # 准备选项
            ydl_opts = {
                'format': 'best',
                'dumpjson': True,
                'ignoreerrors': True,
                **cookies_opts  # 添加 cookies 配置
            }
            
            if is_single_video:
                # 单个视频：即使URL包含播放列表参数，也只下载该视频
                ydl_opts['noplaylist'] = True
                logger.info(f'Detected single video URL, downloading only this video: {u}')
            else:
                # 播放列表：限制下载数量
                ydl_opts['playlistend'] = num_videos
                logger.info(f'Detected playlist URL, downloading up to {num_videos} videos')
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(u, download=False)
            
            # 检查 result 是否为 None（下载失败时可能返回 None）
            if result is None:
                logger.error(f'Failed to extract info from {u}: extract_info returned None (可能是 YouTube bot 验证失败)')
                logger.error('建议：1. 设置环境变量 YOUTUBE_COOKIES_FILE 指向 cookies.txt 文件')
                logger.error('      2. 或使用浏览器扩展导出 cookies')
                continue
            
            # 检查是否是播放列表
            if 'entries' in result:
                # 播放列表
                entries = result.get('entries', [])
                if entries:
                    video_info_list.extend([v for v in entries if v])
                else:
                    logger.warning(f'播放列表为空: {u}')
            else:
                # 单个视频
                video_info_list.append(result)
                    
        except yt_dlp.utils.ExtractorError as e:
            error_msg = str(e)
            if 'Sign in to confirm' in error_msg or 'bot' in error_msg.lower():
                logger.error(f'YouTube bot 验证失败: {u}')
                logger.error('解决方案：')
                logger.error('1. 设置环境变量: export YOUTUBE_COOKIES_FILE=/path/to/cookies.txt')
                logger.error('2. 导出 cookies：使用浏览器扩展（如 "Get cookies.txt LOCALLY"）')
                logger.error('3. 或使用 yt-dlp 的 --cookies-from-browser 选项')
            else:
                logger.error(f'Failed to extract info from {u}: {e}')
        except Exception as e:
            logger.error(f'Failed to extract info from {u}: {e}')
            import traceback
            logger.debug(traceback.format_exc())

    # 下载视频
    logger.info(f'Downloading {len(video_info_list)} video(s)')
    if video_info_list:
        download_videos(video_info_list, folder_path, resolution)
    else:
        logger.warning('没有可下载的视频（可能都因为 bot 验证失败）')


if __name__ == '__main__':
    # 示例用法
    test_url = 'https://www.youtube.com/watch?v=3LPJfIKxwWc'
    test_folder = 'videos'
    download_from_url(test_url, test_folder)
