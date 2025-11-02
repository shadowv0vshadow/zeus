"""Bilibili 上传模块

该模块提供视频上传到 Bilibili 的功能
使用 bilibili-toolman 进行同步上传
"""

import json
import os
import time
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from loguru import logger

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available, cannot auto-generate cover images")

# 导入 bilibili-toolman 同步上传库
try:
    from bilibili_toolman.bilisession.web import BiliSession
    from bilibili_toolman.bilisession.common.submission import Submission
except ImportError:
    logger.error("bilibili-toolman 未安装，请运行: pip install bilibili-toolman")
    raise ImportError("bilibili-toolman 未安装，请运行: pip install bilibili-toolman")

load_dotenv()


def bili_login() -> BiliSession:
    """登录 Bilibili

    Returns:
        BiliSession 对象

    Raises:
        Exception: 登录失败时抛出异常
    """
    sessdata = os.getenv("BILI_SESSDATA")
    bili_jct = os.getenv("BILI_BILI_JCT")

    if not sessdata or not bili_jct:
        raise Exception("BILI_SESSDATA 和 BILI_BILI_JCT 环境变量未设置")

    try:
        session = BiliSession(f"SESSDATA={sessdata};bili_jct={bili_jct}")
        return session
    except Exception as e:
        logger.error(f"Bilibili 登录失败: {e}")
        raise Exception("Bilibili 登录失败，请检查 SESSDATA 和 bili_jct 是否正确")


def upload_video(folder: str) -> bool:
    """上传视频到 Bilibili（使用 bilibili-toolman）

    Args:
        folder: 视频文件所在文件夹

    Returns:
        是否上传成功

    Raises:
        Exception: 上传失败时抛出异常
    """
    submission_result_path = os.path.join(folder, "bilibili.json")

    # 检查是否已经上传
    if os.path.exists(submission_result_path):
        with open(submission_result_path, "r", encoding="utf-8") as f:
            submission_result = json.load(f)
        # 检查是否有 bvid（新格式）或 code=0（旧格式）
        if submission_result.get("bvid"):
            logger.info(f"视频已上传过 (bvid: {submission_result['bvid']})，跳过")
            return True
        if submission_result.get("results", [{}])[0].get("code") == 0:
            logger.info("视频已上传过，跳过")
            return True

    # 检查必需的文件
    video_path = os.path.join(folder, "video.mp4")
    cover_path = os.path.join(folder, "video.png")
    summary_path = os.path.join(folder, "summary.json")
    info_path = os.path.join(folder, "download.info.json")

    # 必需文件：视频和摘要
    required_files = [video_path, summary_path]
    for path in required_files:
        if not os.path.exists(path):
            logger.error(f"必需文件不存在: {path}")
            return False

    # 封面图片处理：如果不存在则尝试生成或使用缩略图
    if not os.path.exists(cover_path):
        # 尝试使用下载的缩略图
        possible_covers = [
            os.path.join(folder, "download.webp"),
            os.path.join(folder, "download.jpg"),
            os.path.join(folder, "download.png"),
        ]
        found_cover = False
        for possible_cover in possible_covers:
            if os.path.exists(possible_cover):
                # 如果有 PIL，尝试将缩略图转换为标准的 video.png
                if PIL_AVAILABLE:
                    try:
                        with Image.open(possible_cover) as img:
                            # 转换为 RGB（如果是 RGBA）
                            if img.mode == "RGBA":
                                img = img.convert("RGB")

                            # 计算尺寸，保持宽高比，目标尺寸 1280x960
                            size = (1280, 960)
                            img_ratio = img.width / img.height
                            target_ratio = size[0] / size[1]

                            if img_ratio < target_ratio:
                                new_height = size[1]
                                new_width = int(new_height * img_ratio)
                            else:
                                new_width = size[0]
                                new_height = int(new_width / img_ratio)

                            # 调整大小
                            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                            # 创建黑色背景
                            new_img = Image.new("RGB", size, "black")

                            # 居中粘贴
                            x_offset = (size[0] - new_width) // 2
                            y_offset = (size[1] - new_height) // 2
                            new_img.paste(img, (x_offset, y_offset))

                            # 保存为 video.png
                            new_img.save(cover_path)
                            found_cover = True
                            break
                    except Exception as e:
                        logger.warning(f"缩略图转换失败: {e}")

                # 如果没有 PIL 或转换失败，直接使用缩略图
                if not found_cover:
                    cover_path = possible_cover
                    found_cover = True
                    break

    # 加载摘要数据
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 清理标题和摘要
    summary["title"] = summary.get("title", "").replace("视频标题：", "").strip()
    summary["summary"] = summary.get("summary", "").replace("视频摘要：", "").replace("视频简介：", "").strip()
    author = summary.get("author", "Unknown")

    # 加载原视频信息
    if not os.path.exists(info_path):
        data = {"title": "", "webpage_url": ""}
    else:
        with open(info_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    # 构建标题（Bilibili 限制最多 80 字）
    title_template = f"【中配】{summary['title']} - {author}"
    if len(title_template) > 80:
        # 如果标题过长，截断原始标题，保留前缀和后缀
        max_summary_length = 80 - len(f"【中配】 - {author}")
        if max_summary_length > 0:
            truncated_summary = summary["title"][:max_summary_length]
            title = f"【中配】{truncated_summary} - {author}"
        else:
            # 如果连前缀和后缀都超过 80 字，只保留前缀和后缀
            title = f"【中配】 - {author}"[:80]
        logger.warning(f"标题过长 ({len(title_template)} 字)，已截断为 {len(title)} 字")
    else:
        title = title_template

    webpage_url = data.get("webpage_url", "")
    description = (
        f"{data.get('title', '')}\n"
        f"{summary['summary']}\n\n"
        f"项目地址：https://github.com/liuzhao1225/YouDub-webui\n"
        f"YouDub 是一个开创性的开源工具，旨在将 YouTube 和其他平台上的高质量视频"
        f"翻译和配音成中文版本。该工具结合了最新的 AI 技术，包括语音识别、"
        f"大型语言模型翻译，以及 AI 声音克隆技术，提供与原视频相似的中文配音，"
        f"为中文用户提供卓越的观看体验。"
    )

    raw_tags = summary.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []

    # 构建标签列表（Bilibili 限制最多 10 个标签，每个标签最多 20 字）
    base_tags = ["YouDub", author, "AI", "中文配音", "科学", "科普"]
    # 合并基础标签和自定义标签，去重
    all_tags = base_tags + raw_tags
    # 去重并截断每个标签到 20 字，然后限制总数最多 10 个
    final_tags = list(dict.fromkeys([tag[:20] for tag in all_tags]))[:10]

    if len(all_tags) > 10:
        logger.warning(f"标签数量超过限制，已截断为 10 个")

    # 登录 Bilibili
    session = bili_login()

    # 提交视频（最多重试 5 次）
    max_retries = 5
    for retry in range(max_retries):
        try:
            logger.info(f"上传视频 (尝试 {retry + 1}/{max_retries})")

            # 上传视频文件
            try:
                upload_result = session.UploadVideo(video_path)

                # 处理返回值
                if isinstance(upload_result, tuple):
                    video_endpoint, upload_info = upload_result
                elif isinstance(upload_result, str):
                    if upload_result == "OK":
                        raise Exception("UploadVideo 返回 'OK'，无法获取 video_endpoint")
                    video_endpoint = upload_result
                else:
                    video_endpoint = upload_result

                if not video_endpoint:
                    raise Exception("无法获取 video_endpoint，视频上传可能失败")

            except KeyError as ke:
                logger.error(f"上传失败: KeyError: {ke}")
                raise Exception(f"上传失败 (KeyError: {ke})，请检查网络和 Cookie")
            except Exception as upload_error:
                logger.error(f"上传失败: {upload_error}")
                raise

            # 创建投稿对象
            submission = Submission(title=title, desc=description)

            # 添加视频
            submission.videos.append(Submission(title=title, video_endpoint=video_endpoint))

            # 上传并设置封面（如果存在）
            if os.path.exists(cover_path):
                try:
                    cover_url = session.UploadCover(cover_path)
                    submission.cover_url = cover_url
                except Exception as cover_error:
                    logger.warning(f"封面上传失败，继续使用无封面上传: {cover_error}")

            # 设置标签
            for tag in final_tags:
                submission.tags.append(tag)

            # 设置分区（科普类）
            submission.thread = 201
            submission.copyright = submission.COPYRIGHT_REUPLOAD
            submission.source = webpage_url

            # 提交投稿
            response = session.SubmitSubmission(submission, seperate_parts=False)

            # 检查返回结果
            results = response.get("results", [])
            if not results:
                raise Exception("Submission 返回结果为空")

            first_result = results[0]
            code = first_result.get("code")

            if code != 0:
                error_message = first_result.get("message", "未知错误")
                raise Exception(f"投稿失败: {error_message}")

            bvid = first_result.get("bvid", "未知")
            logger.info(f"✅ 投稿成功！BVID: {bvid}")

            # 保存结果
            with open(submission_result_path, "w", encoding="utf-8") as f:
                json.dump(response, f, ensure_ascii=False, indent=4)

            return True

        except Exception as e:
            logger.error(f"上传失败 (尝试 {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                wait_time = 10 * (retry + 1)
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)

    raise Exception(f"上传失败，已重试 {max_retries} 次")


def get_all_upload_projects(root_folder: str = "videos") -> list:
    """获取所有可以上传的项目（包含 video.mp4 的文件夹）

    Args:
        root_folder: 根文件夹路径

    Returns:
        项目列表，每个项目包含 path, name, title 等信息
    """
    projects = []

    if not os.path.exists(root_folder):
        return projects

    # 扫描所有包含 video.mp4 的文件夹
    for root, dirs, files in os.walk(root_folder):
        if "video.mp4" in files:
            # 获取项目名称（相对路径）
            project_name = os.path.relpath(root, root_folder)

            # 获取项目标题（从 download.info.json 或 summary.json）
            title = project_name
            info_path = os.path.join(root, "download.info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        info = json.load(f)
                        title = info.get("title", project_name)
                except:
                    pass
            else:
                # 如果没有 download.info.json，尝试从 summary.json 读取标题
                summary_path = os.path.join(root, "summary.json")
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary = json.load(f)
                            title = summary.get("title", project_name).replace("视频标题：", "").strip()
                    except:
                        pass

            # 检查上传状态
            bilibili_json_path = os.path.join(root, "bilibili.json")
            upload_status = "未上传"
            if os.path.exists(bilibili_json_path):
                try:
                    with open(bilibili_json_path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                        # 检查新格式（bvid）或旧格式（code）
                        if result.get("bvid"):
                            upload_status = "已上传"
                        elif result.get("results", [{}])[0].get("code") == 0:
                            upload_status = "已上传"
                except:
                    pass

            projects.append(
                {
                    "path": root,
                    "name": project_name,
                    "title": title,
                    "upload_status": upload_status,
                }
            )

    return sorted(projects, key=lambda x: x["name"])


def upload_all_videos_under_folder(folder: str) -> str:
    """上传指定文件夹下的所有视频

    Args:
        folder: 根文件夹路径

    Returns:
        处理结果描述
    """
    logger.info(f"开始上传文件夹下的所有视频: {folder}")
    uploaded_count = 0
    failed_count = 0

    for root, _, files in os.walk(folder):
        if "video.mp4" in files:
            try:
                upload_video(root)
                uploaded_count += 1
            except Exception as e:
                logger.error(f"上传失败: {e}")
                failed_count += 1

    result_msg = f"上传完成。成功: {uploaded_count}, 失败: {failed_count}"
    logger.info(result_msg)
    return result_msg


if __name__ == "__main__":
    # 示例用法
    test_folder = r"videos\The Game Theorists\20210522 Game Theory What Level is Ashs Pikachu Pokemon"
    upload_all_videos_under_folder(test_folder)
