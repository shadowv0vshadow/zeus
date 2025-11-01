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
    
    # 优先检查环境变量中的 cookies 文件路径
    cookies_file = os.getenv('YOUTUBE_COOKIES_FILE')
    if cookies_file and os.path.exists(cookies_file):
        cookies_opts['cookies'] = cookies_file
        logger.info(f'✓ 使用自定义 cookies 文件: {cookies_file}')
        return cookies_opts
    
    # 尝试从浏览器自动提取 cookies（使用 yt-dlp 的 cookiesfrombrowser）
    try:
        system = platform.system()
        browsers_to_try = []
        
        if system == 'Darwin':  # macOS
            browsers_to_try = ['chrome', 'safari', 'firefox', 'edge']
        elif system == 'Linux':
            browsers_to_try = ['chrome', 'chromium', 'firefox', 'opera', 'edge']
        elif system == 'Windows':
            browsers_to_try = ['chrome', 'edge', 'firefox', 'opera']
        
        # 尝试每个浏览器（直接使用，yt-dlp 会自动处理）
        for browser in browsers_to_try:
            try:
                # 直接设置 cookiesfrombrowser，yt-dlp 会自动查找浏览器
                cookies_opts['cookiesfrombrowser'] = (browser,)
                logger.info(f'尝试使用 {browser} 浏览器的 cookies...')
                # 直接返回，让 yt-dlp 在实际使用时处理
                logger.info(f'✓ 将尝试使用 {browser} 浏览器的 cookies')
                return cookies_opts
            except Exception as e:
                logger.debug(f'{browser} 浏览器不可用: {e}')
                continue
        
        # 如果所有浏览器都不可用，尝试检查 Chrome cookies 文件是否存在
        system = platform.system()
        chrome_paths = []
        if system == 'Darwin':
            chrome_paths = [
                os.path.expanduser('~/Library/Application Support/Google/Chrome/Default/Cookies'),
                os.path.expanduser('~/Library/Application Support/Google/Chrome/Profile 1/Cookies')
            ]
        elif system == 'Linux':
            chrome_paths = [
                os.path.expanduser('~/.config/google-chrome/Default/Cookies'),
                os.path.expanduser('~/.config/google-chrome/Profile 1/Cookies'),
                os.path.expanduser('~/.config/chromium/Default/Cookies')
            ]
        elif system == 'Windows':
            chrome_paths = [
                os.path.expanduser('~/AppData/Local/Google/Chrome/User Data/Default/Cookies'),
                os.path.expanduser('~/AppData/Local/Google/Chrome/User Data/Profile 1/Cookies')
            ]
        
        for chrome_path in chrome_paths:
            if os.path.exists(chrome_path):
                cookies_opts['cookiesfrombrowser'] = ('chrome',)
                logger.info(f'✓ 检测到 Chrome cookies 文件，将使用 Chrome cookies')
                return cookies_opts
                
    except Exception as e:
        logger.debug(f'无法自动检测浏览器 cookies: {e}')
    
    # 如果都没有，返回空字典（不使用 cookies）
    logger.warning('⚠ 未检测到浏览器 cookies，可能会遇到 YouTube bot 验证')
    logger.warning('解决方案：')
    logger.warning('1. 设置环境变量: export YOUTUBE_COOKIES_FILE=/path/to/cookies.txt')
    logger.warning('2. 使用浏览器扩展导出 cookies（推荐）：')
    logger.warning('   - Chrome/Edge: 安装 "Get cookies.txt LOCALLY" 扩展')
    logger.warning('   - 登录 YouTube -> 点击扩展 -> 导出 cookies.txt')
    logger.warning('   - 然后运行: export YOUTUBE_COOKIES_FILE=/path/to/cookies.txt')
    logger.warning('3. 查看详细指南: cat COOKIES_GUIDE.md')
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
                logger.error(f'❌ YouTube bot 验证失败: {u}')
                logger.error('')
                logger.error('当前 cookies 配置：')
                if 'cookiesfrombrowser' in cookies_opts:
                    browser = cookies_opts['cookiesfrombrowser'][0] if isinstance(cookies_opts['cookiesfrombrowser'], (list, tuple)) else str(cookies_opts['cookiesfrombrowser'])
                    logger.error(f'  - 使用浏览器: {browser} (cookiesfrombrowser)')
                elif 'cookies' in cookies_opts:
                    logger.error(f'  - Cookies 文件: {cookies_opts["cookies"]}')
                else:
                    logger.error('  - 未配置 cookies')
                logger.error('')
                logger.error('解决方案：')
                logger.error('1. 【推荐】使用浏览器扩展导出 cookies.txt:')
                logger.error('   - Chrome/Edge: 安装 "Get cookies.txt LOCALLY" 扩展')
                logger.error('   - Firefox: 安装 "cookies.txt" 扩展')
                logger.error('   - 登录 YouTube -> 点击扩展图标 -> 导出 cookies.txt')
                logger.error('   - 设置: export YOUTUBE_COOKIES_FILE=/path/to/cookies.txt')
                logger.error('')
                logger.error('2. 或手动设置环境变量:')
                logger.error('   export YOUTUBE_COOKIES_FILE=/path/to/cookies.txt')
                logger.error('')
                logger.error('3. 查看详细指南: cat COOKIES_GUIDE.md')
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
