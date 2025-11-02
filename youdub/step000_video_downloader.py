import os
import re
import subprocess
import shutil
import json
from loguru import logger
import yt_dlp
import io


def prepare_cookies(input_path=None, output_path=None):
    """
    预处理 cookies 文件，去掉 BOM、转换换行符与编码格式。
    适用于从浏览器导出的 Netscape 格式 cookies 文件。
    参数：
        input_path (str): 原始 cookies 文件路径（人工导出版本），如果为 None，则从环境变量读取
        output_path (str): 转换后供 yt-dlp 使用的 cookies 文件路径，如果为 None，则使用 input_path
    返回：
        str: 输出文件路径（便于后续传递给 yt-dlp），失败返回 None
    """
    # 如果没有指定输入路径，尝试从环境变量读取
    if input_path is None:
        input_path = os.getenv("YOUTUBE_COOKIES_FILE")
        if not input_path:
            logger.warning("未指定 cookies 文件路径，且环境变量 YOUTUBE_COOKIES_FILE 未设置")
            return None

    # 展开用户目录（~）
    input_path = os.path.expanduser(input_path)
    input_path = os.path.abspath(input_path)

    # 如果没有指定输出路径，使用输入路径（原地处理）
    if output_path is None:
        output_path = input_path
    else:
        output_path = os.path.expanduser(output_path)
        output_path = os.path.abspath(output_path)

    if not os.path.exists(input_path):
        logger.error(f"Cookies 文件不存在: {input_path}")
        return None

    try:
        with io.open(input_path, "r", encoding="utf-8-sig") as f:
            content = f.read().replace("\r\n", "\n")  # 转为 LF
        with io.open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Cookies 文件已转换并保存至: {output_path}")
        return output_path
    except Exception as e:
        logger.exception(f"处理 cookies 文件时出错: {e}")
        return None


def sanitize_title(title):
    # Only keep numbers, letters, Chinese characters, and spaces
    title = re.sub(r"[^\w\u4e00-\u9fff \d_-]", "", title)
    # Replace multiple spaces with a single space
    title = re.sub(r"\s+", " ", title)
    return title


def get_target_folder(info, folder_path):
    sanitized_title = sanitize_title(info["title"])
    sanitized_uploader = sanitize_title(info.get("uploader", "Unknown"))
    upload_date = info.get("upload_date", "Unknown")
    if upload_date == "Unknown":
        return None

    output_folder = os.path.join(folder_path, sanitized_uploader, f"{upload_date} {sanitized_title}")

    return output_folder


def download_single_video(info, folder_path, resolution="1080p"):
    """下载单个视频"""
    sanitized_title = sanitize_title(info.get("title", "Unknown"))
    sanitized_uploader = sanitize_title(info.get("uploader", "Unknown"))
    upload_date = info.get("upload_date", "Unknown")
    if upload_date == "Unknown":
        logger.warning(f"No upload date found for video: {sanitized_title}")
        return None

    output_folder = os.path.join(folder_path, sanitized_uploader, f"{upload_date} {sanitized_title}")
    if os.path.exists(os.path.join(output_folder, "download.mp4")):
        logger.info(f"Video already downloaded in {output_folder}")
        return output_folder

    # 获取 cookies 文件路径
    cookies_file = _get_cookies_file()

    resolution_value = resolution.replace("p", "")
    ydl_opts = {
        "format": f"bestvideo[ext=mp4][height<={resolution_value}]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "writeinfojson": True,
        "writethumbnail": True,
        "outtmpl": os.path.join(output_folder, "download"),
        "ignoreerrors": True,
    }

    # 添加 cookies（使用 cookiefile，这是 yt-dlp 支持的参数名）
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file
        logger.info(f"使用 cookies 文件: {cookies_file}")

    try:
        webpage_url = info.get("webpage_url") or info.get("url")
        if not webpage_url:
            logger.error(f"No webpage_url found in video info: {sanitized_title}")
            return None

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([webpage_url])
        logger.info(f"Video downloaded in {output_folder}")
        return output_folder
    except Exception as e:
        logger.error(f"Failed to download video {sanitized_title}: {e}")
        return None


def download_videos(info_list, folder_path, resolution="1080p"):
    for info in info_list:
        download_single_video(info, folder_path, resolution)


def _get_cookies_file():
    """获取 cookies 文件路径

    Returns:
        str: cookies 文件路径，如果不存在则返回 None
    """
    # 优先从环境变量读取
    cookies_file = os.getenv("YOUTUBE_COOKIES_FILE")
    if cookies_file:
        cookies_file = os.path.expanduser(cookies_file)
        cookies_file = os.path.abspath(cookies_file)
        if os.path.exists(cookies_file):
            logger.info(f"从环境变量读取 cookies 文件: {cookies_file}")
            # 预处理 cookies（去掉 BOM、转换换行符）
            prepared = prepare_cookies(cookies_file, cookies_file)
            return prepared if prepared else cookies_file
        else:
            logger.warning(f"环境变量指定的 cookies 文件不存在: {cookies_file}")

    # 尝试默认路径
    default_paths = [
        os.path.abspath("cookies.txt"),
        os.path.abspath("../cookies.txt"),
        os.path.expanduser("~/cookies.txt"),
    ]

    for path in default_paths:
        if os.path.exists(path):
            logger.info(f"找到默认 cookies 文件: {path}")
            prepared = prepare_cookies(path, path)
            return prepared if prepared else path

    logger.warning("未找到 cookies 文件，可能会遇到 YouTube bot 验证")
    return None


def get_info_list_from_url(url, num_videos):
    """从 URL 获取视频信息列表（生成器）"""
    if isinstance(url, str):
        url = [url]

    # 获取 cookies 文件
    cookies_file = _get_cookies_file()

    ydl_opts = {
        "format": "best",
        "dumpjson": True,
        "playlistend": num_videos,
        "ignoreerrors": True,
    }

    # 添加 cookies
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for u in url:
            try:
                result = ydl.extract_info(u, download=False)
                if result is None:
                    logger.error(f"extract_info 返回 None for URL: {u}")
                    continue

                if "entries" in result:
                    # Playlist
                    for video_info in result.get("entries", []):
                        if video_info:
                            yield video_info
                else:
                    # Single video
                    yield result
            except Exception as e:
                logger.error(f"Failed to extract info from {u}: {e}")
                # 如果 Python API 失败，尝试命令行
                try:
                    result = _extract_info_via_cli(u, cookies_file)
                    if result:
                        if "entries" in result:
                            for video_info in result.get("entries", []):
                                if video_info:
                                    yield video_info
                        else:
                            yield result
                except Exception as cli_error:
                    logger.error(f"命令行方式也失败: {cli_error}")


def _extract_info_via_cli(url, cookies_file=None):
    """使用命令行方式提取视频信息（作为 Python API 的备用方案）

    Args:
        url: 视频 URL
        cookies_file: cookies 文件路径

    Returns:
        视频信息字典，失败返回 None
    """
    # 检查 yt-dlp 命令是否可用
    ytdlp_cmd = shutil.which("yt-dlp")
    if not ytdlp_cmd:
        ytdlp_cmd = shutil.which("youtube-dl")

    if not ytdlp_cmd:
        logger.error("❌ 未找到 yt-dlp 或 youtube-dl 命令")
        return None

    logger.info(f"使用命令行工具提取信息: {ytdlp_cmd}")

    # 构建命令行参数
    cmd = [ytdlp_cmd, "--dump-json", "--no-warnings", "--no-playlist"]

    # 添加 cookies
    if cookies_file and os.path.exists(cookies_file):
        cmd.extend(["--cookies", cookies_file])
        logger.info(f"命令行使用 cookies: {cookies_file}")

    # 添加 URL
    cmd.append(url)

    # 执行命令
    try:
        logger.debug(f'执行命令: {" ".join(cmd)}')
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)  # 5分钟超时

        # 解析 JSON 输出
        output_lines = result.stdout.strip().split("\n")
        for line in output_lines:
            if line.strip():
                try:
                    video_info = json.loads(line)
                    logger.info(f'✓ 命令行成功提取视频信息: {video_info.get("title", "Unknown")}')
                    return video_info
                except json.JSONDecodeError:
                    continue

        logger.error("❌ 命令行输出无法解析为 JSON")
        return None

    except subprocess.TimeoutExpired:
        logger.error("❌ 命令行执行超时")
        return None
    except subprocess.CalledProcessError as e:
        error_output = e.stderr or e.stdout or str(e)
        logger.error(f"❌ 命令行执行失败: {error_output}")
        return None
    except Exception as e:
        logger.error(f"❌ 命令行执行异常: {e}")
        return None


def download_from_url(url, folder_path, resolution="1080p", num_videos=5):
    """从 URL 下载视频"""
    if isinstance(url, str):
        url = [url]

    logger.info(f"Fetching video information from {len(url)} URL(s)")

    # 获取 cookies 文件
    cookies_file = _get_cookies_file()

    # 检测URL类型：如果包含 watch?v=，视为单个视频；否则视为播放列表
    video_info_list = []
    for u in url:
        is_single_video = "watch?v=" in u or "/watch/" in u

        ydl_opts = {
            "format": "best",
            "dumpjson": True,
            "ignoreerrors": False,  # 改为 False 以便捕获真实错误
        }

        # 添加 cookies
        if cookies_file:
            ydl_opts["cookiefile"] = cookies_file
            logger.info(f"使用 cookies 文件: {cookies_file}")

        # 添加播放列表选项
        if is_single_video:
            ydl_opts["noplaylist"] = True
            logger.info(f"检测到单个视频 URL: {u}")
        else:
            ydl_opts["playlistend"] = num_videos
            logger.info(f"检测到播放列表 URL，将下载最多 {num_videos} 个视频: {u}")

        result = None
        try:
            # 先尝试 Python API
            logger.info("尝试使用 Python API 提取视频信息...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(u, download=False)

            if result is None:
                raise ValueError("extract_info returned None")

            logger.info(f"✓ Python API 成功提取视频信息")

        except (yt_dlp.utils.ExtractorError, ValueError, Exception) as api_error:
            error_msg = str(api_error)
            logger.warning(f"⚠ Python API 提取失败: {error_msg}")

            # 如果是 bot 验证错误，尝试命令行
            if "Sign in to confirm" in error_msg or "bot" in error_msg.lower():
                logger.info("检测到 bot 验证错误，切换到命令行方式...")
                result = _extract_info_via_cli(u, cookies_file)
            else:
                # 其他错误也尝试命令行
                logger.info("尝试使用命令行方式作为备用方案...")
                result = _extract_info_via_cli(u, cookies_file)

        if result is None:
            logger.error(f"❌ 所有方法都失败，无法提取视频信息: {u}")
            continue

        # 处理结果
        if "entries" in result:
            # Playlist
            entries = result.get("entries", [])
            if entries:
                video_info_list.extend([v for v in entries if v])
            else:
                logger.warning(f"播放列表为空: {u}")
        else:
            # Single video
            video_info_list.append(result)

    # 下载视频
    logger.info(f"Downloading {len(video_info_list)} video(s)")
    if video_info_list:
        download_videos(video_info_list, folder_path, resolution)
    else:
        logger.warning("没有可下载的视频（可能都因为 bot 验证失败）")


if __name__ == "__main__":
    # Example usage
    url = "https://www.youtube.com/watch?v=3LPJfIKxwWc"
    folder_path = "videos"
    download_from_url(url, folder_path, resolution="1080p", num_videos=1)
