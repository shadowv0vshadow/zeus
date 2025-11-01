import gradio as gr
import json
import pandas as pd
from youdub.step000_video_downloader import download_from_url
from youdub.step010_demucs_vr import separate_all_audio_under_folder
from youdub.step020_whisperx import transcribe_all_audio_under_folder
from youdub.step030_translation import translate_all_transcript_under_folder
from youdub.step040_tts import (
    generate_all_wavs_under_folder,
    get_all_projects, get_audio_segments, 
    load_audio_config, save_audio_config,
    regenerate_combined_audio
)
from youdub.step050_synthesize_video import synthesize_all_video_under_folder, synthesize_video
from youdub.step060_genrate_info import generate_all_info_under_folder
from youdub.step070_upload_bilibili import upload_all_videos_under_folder
from youdub.do_everything import do_everything
from youdub.step043_tts_aliyun import get_available_voices, DEFAULT_VOICES
import os


def do_everything_wrapper(
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
    whisper_min_speakers=None,
    whisper_max_speakers=None,
    translation_target_language: str = '简体中文',
    translation_model_provider: str = 'openai',
    tts_model: str = 'xtts_v2',
    aliyun_voice_str: str = None,
    subtitles: bool = True,
    speed_up: float = 1.05,
    fps: int = 30,
    target_resolution: str = '1080p',
    max_workers: int = 3,
    max_retries: int = 5,
    auto_upload_video: bool = True
) -> str:
    """包装函数，为 do_everything 提供默认的 xtts_model_name"""
    # 根据 tts_model 自动设置 xtts_model_name
    if tts_model == 'xtts_v1':
        xtts_model_name = 'tts_models/multilingual/multi-dataset/xtts_v1'
    elif tts_model == 'xtts_v2.0.2':
        xtts_model_name = 'tts_models/multilingual/multi-dataset/xtts_v2.0.2'
    else:  # xtts_v2 或其他
        xtts_model_name = 'tts_models/multilingual/multi-dataset/xtts_v2'
    
    # 解析音色字符串（从 "voice_id - description" 格式中提取 voice_id）
    aliyun_voice = None
    if tts_model == 'aliyun':
        if aliyun_voice_str:
            aliyun_voice = aliyun_voice_str.split(' - ')[0] if ' - ' in aliyun_voice_str else aliyun_voice_str
        else:
            aliyun_voice = DEFAULT_VOICES.get('zh', 'longyue_v2')
    
    return do_everything(
        root_folder=root_folder,
        url=url,
        num_videos=num_videos,
        resolution=resolution,
        demucs_model=demucs_model,
        device=device,
        shifts=shifts,
        whisper_model=whisper_model,
        whisper_download_root=whisper_download_root,
        whisper_batch_size=whisper_batch_size,
        whisper_diarization=whisper_diarization,
        whisper_min_speakers=whisper_min_speakers,
        whisper_max_speakers=whisper_max_speakers,
        translation_target_language=translation_target_language,
        translation_model_provider=translation_model_provider,
        tts_model=tts_model,
        xtts_model_name=xtts_model_name,
        aliyun_voice=aliyun_voice if tts_model == 'aliyun' else None,
        subtitles=subtitles,
        speed_up=speed_up,
        fps=fps,
        target_resolution=target_resolution,
        max_workers=max_workers,
        max_retries=max_retries,
        auto_upload_video=auto_upload_video
    )


def update_voice_dropdown(tts_model):
    """根据 TTS 模型选择更新音色下拉框的可见性和选项"""
    if tts_model == 'aliyun':
        # 获取所有中文音色（默认）
        voices = get_available_voices('zh')
        # 格式化为 "音色名称 - 描述" 的格式
        voice_options = [f"{voice_id} - {desc}" for voice_id, desc in voices.items()]
        default_voice_id = DEFAULT_VOICES.get('zh', 'longyue_v2')
        default_value = f"{default_voice_id} - {voices.get(default_voice_id, '')}"
        return gr.Dropdown(choices=voice_options, visible=True, value=default_value)
    else:
        return gr.Dropdown(choices=[], visible=False, value=None)


# 准备音色选项（中文音色）
zh_voices = get_available_voices('zh')
voice_options = [f"{voice_id} - {desc}" for voice_id, desc in zh_voices.items()]
default_voice_value = f"{DEFAULT_VOICES.get('zh', 'longyue_v2')} - {zh_voices.get(DEFAULT_VOICES.get('zh', 'longyue_v2'), '')}"

# 使用 Blocks 来支持动态交互
with gr.Blocks(title='YouDub - 全自动') as do_everything_interface:
    root_folder = gr.Textbox(label='Root Folder', value='videos')
    url = gr.Textbox(label='Video URL', placeholder='Video or Playlist or Channel URL',
                    value='https://www.bilibili.com/list/1263732318')
    num_videos = gr.Slider(minimum=1, maximum=500, step=1, label='Number of videos to download', value=20)
    resolution = gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='Resolution', value='1080p')
    demucs_model = gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'], label='Demucs Model', value='htdemucs_ft')
    device = gr.Radio(['auto', 'cuda', 'cpu'], label='Demucs Device', value='auto')
    shifts = gr.Slider(minimum=0, maximum=10, step=1, label='Number of shifts', value=5)
    whisper_model = gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label='Whisper Model', value='large')
    whisper_download_root = gr.Textbox(label='Whisper Download Root', value='models/ASR/whisper')
    whisper_batch_size = gr.Slider(minimum=1, maximum=128, step=1, label='Whisper Batch Size', value=32)
    whisper_diarization = gr.Checkbox(label='Whisper Diarization', value=True)
    whisper_min_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label='Whisper Min Speakers', value=None)
    whisper_max_speakers = gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label='Whisper Max Speakers', value=None)
    translation_target_language = gr.Dropdown(['简体中文', '繁体中文', 'English', 'Deutsch', 'Français', 'русский'],
                                              label='Translation Target Language', value='简体中文')
    translation_model_provider = gr.Radio(['openai', 'aliyun'], label='Translation Model Provider', value='openai',
                                         info='选择翻译模型提供商：OpenAI 或 阿里云通义大模型')
    tts_model_radio = gr.Radio(['bytedance', 'aliyun', 'xtts_v1', 'xtts_v2', 'xtts_v2.0.2'], label='TTS Model', value='xtts_v2',
                              info='选择语音合成模型：字节跳动、阿里云或 XTTS')
    aliyun_voice_dropdown = gr.Dropdown(choices=voice_options, label='阿里云音色', value=default_voice_value, visible=False,
                                        info='选择阿里云 CosyVoice 音色（仅当 TTS Model 为 aliyun 时有效）')
    subtitles = gr.Checkbox(label='Subtitles', value=True)
    speed_up = gr.Slider(minimum=0.5, maximum=2, step=0.05, label='Speed Up', value=1.05)
    fps = gr.Slider(minimum=1, maximum=60, step=1, label='FPS', value=30)
    target_resolution = gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='Resolution', value='1080p')
    max_workers = gr.Slider(minimum=1, maximum=100, step=1, label='Max Workers', value=1)
    max_retries = gr.Slider(minimum=1, maximum=10, step=1, label='Max Retries', value=3)
    auto_upload_video = gr.Checkbox(label='Auto Upload Video', value=True)
    output = gr.Textbox(label='Output', lines=10)
    
    submit_btn = gr.Button('Submit')
    submit_btn.click(
        fn=do_everything_wrapper,
        inputs=[
            root_folder, url, num_videos, resolution, demucs_model, device, shifts,
            whisper_model, whisper_download_root, whisper_batch_size, whisper_diarization,
            whisper_min_speakers, whisper_max_speakers, translation_target_language,
            translation_model_provider, tts_model_radio, aliyun_voice_dropdown,
            subtitles, speed_up, fps, target_resolution, max_workers, max_retries, auto_upload_video
        ],
        outputs=output
    )
    
    # 添加事件监听，当 TTS Model 改变时更新音色下拉框
    tts_model_radio.change(
        fn=update_voice_dropdown,
        inputs=[tts_model_radio],
        outputs=[aliyun_voice_dropdown]
    )

youtube_interface = gr.Interface(
    fn=download_from_url,
    inputs=[
        gr.Textbox(label='Video URL', placeholder='Video or Playlist or Channel URL',
                   value='https://www.bilibili.com/list/1263732318'),  # Changed 'default' to 'value'
        gr.Textbox(label='Output Folder', value='videos'),  # Changed 'default' to 'value'
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='Resolution', value='1080p'),
        gr.Slider(minimum=1, maximum=100, step=1, label='Number of videos to download', value=5),
    ],
    outputs='text',
    flagging_mode='never',
)

demucs_interface = gr.Interface(
    fn=separate_all_audio_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),  # Changed 'default' to 'value'
        gr.Radio(['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx', 'mdx_extra', 'mdx_q', 'mdx_extra_q', 'SIG'], label='Model', value='htdemucs_ft'),
        gr.Radio(['auto', 'cuda', 'cpu'], label='Device', value='auto'),
        gr.Checkbox(label='Progress Bar in Console', value=True),
        gr.Slider(minimum=0, maximum=10, step=1, label='Number of shifts', value=5),
    ],
    outputs='text',
    flagging_mode='never',
)

# transcribe_all_audio_under_folder(folder, model_name: str = 'large', download_root='models/ASR/whisper', device='auto', batch_size=32)
whisper_inference = gr.Interface(
    fn = transcribe_all_audio_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),  # Changed 'default' to 'value'
        gr.Radio(['large', 'medium', 'small', 'base', 'tiny'], label='Model', value='large'),
        gr.Textbox(label='Download Root', value='models/ASR/whisper'),
        gr.Radio(['auto', 'cuda', 'cpu'], label='Device', value='auto'),
        gr.Slider(minimum=1, maximum=128, step=1, label='Batch Size', value=32),
        gr.Checkbox(label='Diarization', value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 label='Whisper Min Speakers', value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 label='Whisper Max Speakers', value=None),
    ],
    outputs='text',
    flagging_mode='never',
)

def refresh_projects_for_translation(folder):
    """刷新翻译项目列表（查找有transcript.json的项目）"""
    projects = []
    for root, dirs, files in os.walk(folder):
        if 'transcript.json' in files:
            # 有transcript.json就可以编辑翻译（即使已经翻译过）
            translation_path = os.path.join(root, 'translation.json')
            has_translation = os.path.exists(translation_path)
            
            # 尝试获取项目标题
            project_name = os.path.basename(root)
            info_path = os.path.join(root, 'download.info.json')
            title = project_name
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                        title = info.get('title', project_name)
                except:
                    pass
            
            projects.append({
                'name': project_name,
                'path': root,
                'title': title,
                'has_translation': has_translation
            })
    
    project_choices = [f"{p['name']} - {p['title'][:50]}" for p in projects]
    project_paths = {f"{p['name']} - {p['title'][:50]}": p['path'] for p in projects}
    return gr.Dropdown(choices=project_choices, value=project_choices[0] if project_choices else None), project_paths

def load_translation_from_project(project_choice, project_paths_dict):
    """从选定的项目加载翻译文本（原文和翻译）"""
    if not project_choice or project_choice not in project_paths_dict:
        return gr.Dataframe(value=[]), "", "请先选择项目"
    
    project_path = project_paths_dict[project_choice]
    
    # 加载transcript.json（原文）
    transcript_path = os.path.join(project_path, 'transcript.json')
    translation_path = os.path.join(project_path, 'translation.json')
    
    if not os.path.exists(transcript_path):
        return gr.Dataframe(value=[]), "", "项目中没有找到transcript.json"
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except Exception as e:
        return gr.Dataframe(value=[]), "", f"读取transcript.json失败: {str(e)}"
    
    # 检查是否有原声音频文件（用于判断是否可以使用原声）
    wavs_folder = os.path.join(project_path, 'wavs')
    
    # 如果有translation.json，加载翻译；否则使用原文
    if os.path.exists(translation_path):
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                translation_data = json.load(f)
            # 确保translation_data和transcript长度一致
            for i, line in enumerate(transcript):
                if i < len(translation_data):
                    if 'translation' in translation_data[i]:
                        transcript[i]['translation'] = translation_data[i]['translation']
                    elif 'translation' not in transcript[i]:
                        transcript[i]['translation'] = transcript[i].get('text', '')
                    # 读取use_original字段（如果存在）
                    if 'use_original' in translation_data[i]:
                        transcript[i]['use_original'] = translation_data[i]['use_original']
                elif 'translation' not in transcript[i]:
                    transcript[i]['translation'] = transcript[i].get('text', '')
        except:
            pass
    
    # 构建表格数据（包含"使用原声"列）
    table_data = []
    for i, line in enumerate(transcript):
        original_text = line.get('text', '')
        translation_text = line.get('translation', original_text)
        segment_id = f'{i:04d}'
        
        # 检查是否有原声文件
        original_file = os.path.join(wavs_folder, f'{segment_id}_original.wav')
        has_original = os.path.exists(original_file) if os.path.exists(wavs_folder) else False
        
        # 读取use_original字段（默认False）
        use_original = line.get('use_original', False) if has_original else False
        
        table_data.append([
            segment_id,
            original_text[:200] + '...' if len(original_text) > 200 else original_text,
            translation_text[:200] + '...' if len(translation_text) > 200 else translation_text,
            use_original
        ])
    
    project_info = f"项目路径: {project_path}\n片段数: {len(transcript)}\n{'已翻译' if os.path.exists(translation_path) else '未翻译'}"
    
    return gr.Dataframe(value=table_data, headers=["编号", "原文", "翻译", "使用原声"], interactive=True), project_path, project_info

def save_translation_edits(project_path, segments_df):
    """保存编辑后的翻译文本到translation.json"""
    if not project_path or not os.path.exists(project_path):
        return "错误: 项目路径无效"
    
    transcript_path = os.path.join(project_path, 'transcript.json')
    translation_path = os.path.join(project_path, 'translation.json')
    
    if not os.path.exists(transcript_path):
        return "错误: 未找到transcript.json"
    
    # 加载原始transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except Exception as e:
        return f"错误: 读取transcript.json失败: {str(e)}"
    
    # 处理Dataframe格式的数据
    # Gradio API可能发送嵌套的数据结构
    import logging
    logger = logging.getLogger(__name__)
    
    if segments_df is not None:
        # 处理可能的嵌套结构（Gradio API格式）
        # 格式可能是: [null, {"headers": [...], "data": [[...], [...]], "metadata": null}]
        if isinstance(segments_df, list) and len(segments_df) >= 2:
            # 如果第一个元素是null，第二个元素是字典
            if segments_df[0] is None and isinstance(segments_df[1], dict):
                if 'data' in segments_df[1]:
                    segments_df = segments_df[1]['data']
        elif isinstance(segments_df, dict):
            # 如果是字典，尝试提取data字段
            if 'data' in segments_df:
                segments_df = segments_df['data']
                # 如果data是一个列表，且第一个元素是另一个数据结构
                if isinstance(segments_df, list) and len(segments_df) > 0:
                    if isinstance(segments_df[0], dict) and 'data' in segments_df[0]:
                        segments_df = segments_df[0]['data']
        
        if isinstance(segments_df, pd.DataFrame):
            segments_df = segments_df.values.tolist()
        elif hasattr(segments_df, 'tolist'):
            segments_df = segments_df.tolist()
    
    logger.debug(f'segments_df类型: {type(segments_df)}, 长度: {len(segments_df) if segments_df else 0}, 内容示例: {segments_df[:2] if segments_df and len(segments_df) > 2 else segments_df}')
    
    # 创建翻译字典和use_original字典（从表格数据）
    translation_dict = {}
    use_original_dict = {}
    if segments_df and len(segments_df) > 0:
        for row_idx, row in enumerate(segments_df):
            if hasattr(row, 'tolist'):
                row = row.tolist()
            elif not isinstance(row, (list, tuple)):
                logger.warning(f'行{row_idx}格式异常: {type(row)}, 跳过: {row}')
                continue
            
            if len(row) >= 4:
                segment_id = str(row[0]).strip()
                try:
                    segment_num = int(segment_id)
                except ValueError:
                    logger.warning(f'无法解析segment_id: {segment_id}')
                    continue
                
                if segment_num < len(transcript):
                    edited_translation = str(row[2]).strip()  # 翻译文本在第3列（索引2）
                    translation_dict[segment_num] = edited_translation
                    
                    # 读取use_original字段（第4列，索引3）
                    use_original_raw = row[3]
                    if isinstance(use_original_raw, bool):
                        use_original = use_original_raw
                    elif isinstance(use_original_raw, str):
                        use_original = use_original_raw.lower().strip() in ('true', '1', 'yes', '✓', 'checked')
                    elif isinstance(use_original_raw, (int, float)):
                        use_original = bool(use_original_raw)
                    else:
                        use_original = str(use_original_raw).lower().strip() in ('true', '1', 'yes', '✓', 'checked')
                    
                    use_original_dict[segment_num] = use_original
                    logger.debug(f'片段{segment_num}: translation="{edited_translation[:30]}...", use_original={use_original}')
                else:
                    logger.warning(f'片段编号{segment_num}超出范围(总共{len(transcript)}个片段)')
            else:
                logger.warning(f'行{row_idx}列数不足: {len(row)} < 4')
    
    logger.info(f'解析结果: translation_dict={len(translation_dict)}个, use_original_dict={len(use_original_dict)}个')
    
    # 更新transcript中的translation和use_original字段
    # 先加载现有的translation.json（如果存在）以保留未编辑的配置
    existing_translation = None
    if os.path.exists(translation_path):
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                existing_translation = json.load(f)
            logger.debug(f'加载现有translation.json，包含{len(existing_translation)}个片段')
        except Exception as e:
            logger.warning(f'加载现有translation.json失败: {e}')
    
    wavs_folder = os.path.join(project_path, 'wavs')
    for i, line in enumerate(transcript):
        # 更新翻译文本
        if i in translation_dict:
            line['translation'] = translation_dict[i]
        elif 'translation' not in line:
            # 如果没有翻译，使用原文
            line['translation'] = line.get('text', '')
        
        # 更新use_original字段
        if i in use_original_dict:
            # 检查是否有原声文件，如果没有则强制为False
            segment_id = f'{i:04d}'
            original_file = os.path.join(wavs_folder, f'{segment_id}_original.wav')
            has_original = os.path.exists(original_file) if os.path.exists(wavs_folder) else False
            
            if has_original:
                line['use_original'] = use_original_dict[i]
                logger.debug(f'片段{i}: 设置use_original={use_original_dict[i]}')
            else:
                line['use_original'] = False
                logger.debug(f'片段{i}: 无原声文件，强制use_original=False')
        else:
            # 如果没有在表格中编辑，保留原有的use_original配置
            if existing_translation and i < len(existing_translation):
                existing_value = existing_translation[i].get('use_original', False)
                line['use_original'] = existing_value
                logger.debug(f'片段{i}: 保留原有use_original={existing_value}')
            elif 'use_original' not in line:
                # 如果完全没有设置，默认为False
                line['use_original'] = False
    
    # 保存到translation.json
    try:
        with open(translation_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        edited_count = len(translation_dict)
        use_original_count = sum(1 for v in use_original_dict.values() if v) if use_original_dict else 0
        return f"✅ 翻译保存成功！\n项目: {os.path.basename(project_path)}\n已编辑片段数: {edited_count}/{len(transcript)}\n使用原声片段数: {use_original_count}/{len(use_original_dict) if use_original_dict else 0}"
    except Exception as e:
        import traceback
        return f"❌ 保存失败: {str(e)}\n{traceback.format_exc()}"

# 使用 Blocks 来支持动态交互
with gr.Blocks(title='YouDub - 字幕翻译') as translation_interface:
    gr.Markdown("# 字幕翻译")
    gr.Markdown("选择项目，查看和编辑翻译文本。可以手动润色翻译结果并保存。")
    
    # 项目选择
    translation_folder = gr.Textbox(label='Folder', value='videos', info='视频文件夹路径')
    translation_refresh_projects_btn = gr.Button('刷新项目列表', variant='secondary', size='sm')
    translation_project_dropdown = gr.Dropdown(
        label='选择项目',
        choices=[],
        value=None,
        info='请先选择要编辑翻译的项目（需要包含transcript.json）'
    )
    translation_project_paths_dict_state = gr.State(value={})  # 存储项目路径字典
    translation_project_path_state = gr.State(value="")  # 存储当前选中的项目路径
    
    # 加载翻译列表按钮
    translation_load_btn = gr.Button('加载翻译文本', variant='primary', size='sm')
    translation_project_info = gr.Textbox(label='项目信息', lines=3, interactive=False, visible=True)
    
    translation_table = gr.Dataframe(
        headers=["编号", "原文", "翻译", "使用原声"],
        label='翻译文本列表（可直接编辑"翻译"列来润色翻译文本，勾选"使用原声"来选择使用原声）',
        interactive=True,
        wrap=True,
        datatype=["str", "str", "str", "bool"],
        value=[]
    )
    
    # 保存翻译按钮
    translation_save_btn = gr.Button('保存翻译', variant='primary', size='lg')
    translation_output = gr.Textbox(label='保存结果', lines=5)
    
    # 当folder改变时自动刷新项目列表
    translation_folder.change(
        fn=refresh_projects_for_translation,
        inputs=[translation_folder],
        outputs=[translation_project_dropdown, translation_project_paths_dict_state]
    )
    
    # 刷新项目列表
    translation_refresh_projects_btn.click(
        fn=refresh_projects_for_translation,
        inputs=[translation_folder],
        outputs=[translation_project_dropdown, translation_project_paths_dict_state]
    )
    
    # 加载翻译文本
    translation_load_btn.click(
        fn=load_translation_from_project,
        inputs=[translation_project_dropdown, translation_project_paths_dict_state],
        outputs=[translation_table, translation_project_path_state, translation_project_info]
    )
    
    # 当项目选择改变时自动加载
    translation_project_dropdown.change(
        fn=load_translation_from_project,
        inputs=[translation_project_dropdown, translation_project_paths_dict_state],
        outputs=[translation_table, translation_project_path_state, translation_project_info]
    )
    
    # 保存翻译
    translation_save_btn.click(
        fn=save_translation_edits,
        inputs=[translation_project_path_state, translation_table],
        outputs=[translation_output]
    )
    
    gr.Markdown("---")
    gr.Markdown("### 自动翻译")
    gr.Markdown("使用AI自动翻译所有未翻译的项目。")
    
    translation_target_language = gr.Dropdown(['简体中文', '繁体中文', 'English', 'Deutsch', 'Français', 'русский'],
                                              label='Target Language', value='简体中文')
    translation_model_provider = gr.Radio(['openai', 'aliyun'], label='Model Provider', value='openai',
                                         info='选择AI模型提供商：OpenAI 或 阿里云通义大模型')
    translation_auto_translate_btn = gr.Button('自动翻译所有项目', variant='primary', size='lg')
    translation_auto_output = gr.Textbox(label='翻译结果', lines=10)
    
    translation_auto_translate_btn.click(
        fn=translate_all_transcript_under_folder,
        inputs=[translation_folder, translation_target_language, translation_model_provider],
        outputs=[translation_auto_output]
    )

def tts_interface_wrapper(folder, tts_model, aliyun_voice_str):
    """TTS 接口包装函数，解析音色字符串"""
    aliyun_voice = None
    if tts_model == 'aliyun' and aliyun_voice_str:
        # 从 "voice_id - description" 格式中提取 voice_id
        aliyun_voice = aliyun_voice_str.split(' - ')[0] if ' - ' in aliyun_voice_str else aliyun_voice_str
    return generate_all_wavs_under_folder(folder, tts_model=tts_model, aliyun_voice=aliyun_voice)

def generate_tts_and_load_segments(folder, tts_model, aliyun_voice_str):
    """生成TTS并加载片段列表"""
    # 生成TTS音频
    tts_result = tts_interface_wrapper(folder, tts_model, aliyun_voice_str)
    
    # 查找所有项目
    projects = get_all_projects(folder)
    
    # 返回TTS结果和项目列表
    project_choices = [f"{p['name']} - {p['title'][:50]}" for p in projects]
    project_paths = {f"{p['name']} - {p['title'][:50]}": p['path'] for p in projects}
    
    return tts_result, gr.Dropdown(choices=project_choices, value=project_choices[0] if project_choices else None), project_paths

def generate_tts_and_regenerate_combined(folder, tts_model, aliyun_voice_str, segments_df):
    """一键生成TTS并生成 audio_combined.wav"""
    # 生成TTS音频
    tts_result = tts_interface_wrapper(folder, tts_model, aliyun_voice_str)
    
    # 如果生成失败，直接返回
    if "失败" in tts_result or "错误" in tts_result:
        return tts_result + "\n\n❌ TTS生成失败，无法继续生成 audio_combined.wav"
    
    # 查找项目（使用folder作为项目路径，如果folder本身就是一个项目）
    project_path = folder
    
    # 检查是否是项目文件夹（有translation.json和wavs）
    translation_path = os.path.join(folder, 'translation.json')
    wavs_path = os.path.join(folder, 'wavs')
    
    if not (os.path.exists(translation_path) and os.path.exists(wavs_path)):
        # 如果不是项目文件夹，查找子文件夹中的项目
        projects = get_all_projects(folder)
        if not projects:
            return tts_result + "\n\n❌ 未找到项目（需要包含translation.json和wavs文件夹）"
        # 使用第一个项目
        project_path = projects[0]['path']
    
    # 从表格数据构建配置
    config = {}
    debug_info = []
    
    # 处理Dataframe格式的数据
    import logging
    logger = logging.getLogger(__name__)
    
    if segments_df is not None:
        if isinstance(segments_df, pd.DataFrame):
            segments_df = segments_df.values.tolist()
        elif hasattr(segments_df, 'tolist'):
            segments_df = segments_df.tolist()
    
    if segments_df and len(segments_df) > 0:
        for row_idx, row in enumerate(segments_df):
            if hasattr(row, 'tolist'):
                row = row.tolist()
            elif not isinstance(row, (list, tuple)):
                continue
                
            if len(row) >= 5:
                segment_id = str(row[0]).strip()
                try:
                    segment_num = int(segment_id)
                    segment_id = f'{segment_num:04d}'
                except ValueError:
                    pass
                
                has_original = row[3] == "✓"
                use_original_raw = row[4]
                
                if isinstance(use_original_raw, bool):
                    use_original = use_original_raw
                elif isinstance(use_original_raw, str):
                    use_original = use_original_raw.lower().strip() in ('true', '1', 'yes', '✓', 'checked')
                elif isinstance(use_original_raw, (int, float)):
                    use_original = bool(use_original_raw)
                else:
                    use_original = str(use_original_raw).lower().strip() in ('true', '1', 'yes', '✓', 'checked')
                
                if not has_original:
                    use_original = False
                
                config[segment_id] = use_original
    
    # 保存配置
    if config:
        save_audio_config(project_path, config)
    
    # 生成 audio_combined.wav
    try:
        regenerate_combined_audio(project_path, config if config else None)
        use_original_count = sum(1 for v in config.values() if v) if config else 0
        total_count = len(config) if config else 0
        
        result = tts_result + f"\n\n✅ audio_combined.wav 生成成功！\n"
        result += f"使用原声的片段: {use_original_count}/{total_count}\n"
        result += f"使用AI配音的片段: {total_count - use_original_count}/{total_count}"
        
        return result
    except Exception as e:
        import traceback
        return tts_result + f"\n\n❌ audio_combined.wav 生成失败: {str(e)}\n{traceback.format_exc()}"

# 视频合成配置界面函数
def refresh_projects_wrapper(root_folder):
    """刷新项目列表"""
    projects = get_all_projects(root_folder)
    project_choices = [f"{p['name']} - {p['title'][:50]}" for p in projects]
    project_paths = {f"{p['name']} - {p['title'][:50]}": p['path'] for p in projects}
    return gr.Dropdown(choices=project_choices, value=None), project_paths

def load_project_segments(project_choice, project_paths_dict):
    """加载项目音频片段信息（从translation.json读取）"""
    if not project_choice or not project_paths_dict or project_choice not in project_paths_dict:
        return "", gr.Dataframe(value=[], headers=["编号", "翻译文本", "有AI配音", "有原声", "使用原声"]), ""
    
    project_path = project_paths_dict[project_choice]
    if not os.path.exists(project_path):
        return f"错误: 项目路径不存在", gr.Dataframe(value=[]), ""
    
    segments = get_audio_segments(project_path)
    config = load_audio_config(project_path)
    
    if not segments:
        return f"项目路径: {project_path}\n音频片段数: 0\n\n未找到translation.json或片段数据", gr.Dataframe(value=[]), project_path
    
    # 构建表格数据（包含复选框列）
    table_data = []
    for segment in segments:
        segment_id = segment['id']
        has_ai_voice = segment['has_ai_voice']
        has_original = segment['has_original']
        use_original = config.get(segment_id, False)
        
        # 表格数据：编号、翻译文本、有AI配音、有原声、使用原声（布尔值，用于Dataframe显示）
        table_data.append([
            segment_id,
            segment['translation'],
            "✓" if has_ai_voice else "✗",
            "✓" if has_original else "✗",
            use_original if has_original else False  # 如果没有原声，强制为False
        ])
    
    ai_voice_count = sum(1 for s in segments if s['has_ai_voice'])
    original_count = sum(1 for s in segments if s['has_original'])
    project_info = f"项目路径: {project_path}\n音频片段数: {len(segments)}\n有AI配音: {ai_voice_count}\n有原声: {original_count}\n\n说明：勾选'使用原声'表示使用原声（0000_original.wav），未勾选使用AI配音（0000.wav）"
    
    return project_info, gr.Dataframe(value=table_data, headers=["编号", "翻译文本", "有AI配音", "有原声", "使用原声"], interactive=True), project_path

def save_config_and_regenerate_audio_wrapper(project_path, segments_df):
    """保存配置并重新生成 audio_combined.wav"""
    if not project_path or not os.path.exists(project_path):
        return "错误: 项目路径无效"
    
    # 从表格数据构建配置
    config = {}
    debug_info = []  # 用于调试
    
    # 处理Dataframe格式的数据（Gradio返回的是pandas DataFrame）
    import logging
    logger = logging.getLogger(__name__)
    
    # 检查是否是DataFrame，如果是则转换为列表
    if segments_df is not None:
        if isinstance(segments_df, pd.DataFrame):
            # 将DataFrame转换为列表的列表
            segments_df = segments_df.values.tolist()
            logger.debug(f'转换DataFrame为列表，行数: {len(segments_df)}')
        elif hasattr(segments_df, 'tolist'):
            # 如果是numpy数组或其他可转换类型
            segments_df = segments_df.tolist()
    
    if segments_df and len(segments_df) > 0:
        logger.debug(f'segments_df类型: {type(segments_df)}, 长度: {len(segments_df)}')
        if len(segments_df) > 0:
            logger.debug(f'第一行数据: {segments_df[0]}, 类型: {type(segments_df[0])}')
        
        for row_idx, row in enumerate(segments_df):
            # 处理可能的DataFrame或列表格式
            if hasattr(row, 'tolist'):
                row = row.tolist()
            elif not isinstance(row, (list, tuple)):
                logger.warning(f'行{row_idx}格式异常: {type(row)}, 跳过')
                continue
                
            if len(row) >= 3:
                segment_id = str(row[0]).strip()  # 编号（确保是字符串，去除空格）
                # 确保 segment_id 是4位数字格式（如 "0011" 而不是 "11"）
                original_segment_id = segment_id
                try:
                    segment_num = int(segment_id)
                    segment_id = f'{segment_num:04d}'  # 格式化为 0011, 0025 等
                except ValueError:
                    pass  # 如果转换失败，使用原始值
                
                # 读取使用原声的值（第3列，索引2）
                use_original_raw = row[2]
                
                # 处理不同类型的布尔值（True/False, "True"/"False", 1/0等）
                if isinstance(use_original_raw, bool):
                    use_original = use_original_raw
                elif isinstance(use_original_raw, str):
                    use_original = use_original_raw.lower().strip() in ('true', '1', 'yes', '✓', 'checked')
                elif isinstance(use_original_raw, (int, float)):
                    use_original = bool(use_original_raw)
                else:
                    # 对于其他类型，尝试转换为字符串再判断
                    use_original = str(use_original_raw).lower().strip() in ('true', '1', 'yes', '✓', 'checked')
                
                config[segment_id] = use_original
                debug_info.append(f"  行{row_idx}: {original_segment_id}->{segment_id}, use_original_raw={repr(use_original_raw)} ({type(use_original_raw).__name__}), use_original={use_original}")
    
    # 如果没有表格数据，尝试从已有配置加载
    if not config:
        config = load_audio_config(project_path)
        # 如果没有配置，获取所有片段并默认全部AI配音
        segments = get_audio_segments(project_path)
        if segments:
            for segment in segments:
                segment_id = segment['id']
                config[segment_id] = False
    
    # 保存配置
    save_audio_config(project_path, config)
    
    # 调试：显示配置内容
    use_original_segments = [k for k, v in config.items() if v]
    config_debug = f"\n配置详情:\n" + "\n".join([f"  片段 {k}: {'原声' if v else 'AI配音'}" for k, v in sorted(config.items())])
    if debug_info:
        config_debug += "\n\n解析详情:\n" + "\n".join(debug_info)
    
    # 打印传递给 regenerate_combined_audio 的配置（用于调试）
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f'传递给 regenerate_combined_audio 的配置: {config}')
    
    # 重新生成 audio_combined.wav
    try:
        regenerate_combined_audio(project_path, config)
        use_original_count = sum(1 for v in config.values() if v)
        return f"✅ 音频重新生成成功！\n项目: {os.path.basename(project_path)}\n已根据配置重新生成 audio_combined.wav。\n使用原声的片段: {use_original_count}/{len(config)}\n使用AI配音的片段: {len(config) - use_original_count}/{len(config)}\n\n使用原声的片段编号: {', '.join(use_original_segments) if use_original_segments else '无'}" + config_debug
    except Exception as e:
        import traceback
        return f"❌ 音频重新生成失败: {str(e)}\n{traceback.format_exc()}"

def generate_tts_with_config(folder, tts_model, aliyun_voice_str, segments_df):
    """一键生成TTS并生成 audio_combined.wav（合并版本）"""
    # 生成TTS音频
    tts_result = tts_interface_wrapper(folder, tts_model, aliyun_voice_str)
    
    # 如果生成失败，直接返回
    if "失败" in tts_result or "错误" in tts_result:
        return tts_result + "\n\n❌ TTS生成失败，无法继续生成 audio_combined.wav"
    
    # 查找项目
    projects = get_all_projects(folder)
    if not projects:
        return tts_result + "\n\n❌ 未找到项目（需要包含translation.json和wavs文件夹）"
    
    # 使用第一个项目（或者可以改为让用户选择）
    project_path = projects[0]['path']
    
    # 从表格数据构建配置
    config = {}
    
    # 处理Dataframe格式的数据
    import logging
    logger = logging.getLogger(__name__)
    
    if segments_df is not None:
        if isinstance(segments_df, pd.DataFrame):
            segments_df = segments_df.values.tolist()
        elif hasattr(segments_df, 'tolist'):
            segments_df = segments_df.tolist()
    
    if segments_df and len(segments_df) > 0:
        for row_idx, row in enumerate(segments_df):
            if hasattr(row, 'tolist'):
                row = row.tolist()
            elif not isinstance(row, (list, tuple)):
                continue
                
            if len(row) >= 5:
                segment_id = str(row[0]).strip()
                try:
                    segment_num = int(segment_id)
                    segment_id = f'{segment_num:04d}'
                except ValueError:
                    pass
                
                has_original = row[3] == "✓"
                use_original_raw = row[4]
                
                if isinstance(use_original_raw, bool):
                    use_original = use_original_raw
                elif isinstance(use_original_raw, str):
                    use_original = use_original_raw.lower().strip() in ('true', '1', 'yes', '✓', 'checked')
                elif isinstance(use_original_raw, (int, float)):
                    use_original = bool(use_original_raw)
                else:
                    use_original = str(use_original_raw).lower().strip() in ('true', '1', 'yes', '✓', 'checked')
                
                if not has_original:
                    use_original = False
                
                config[segment_id] = use_original
    
    # 如果没有表格数据，默认全部AI配音
    if not config:
        segments = get_audio_segments(project_path)
        if segments:
            for segment in segments:
                segment_id = segment['id']
                config[segment_id] = False
    
    # 保存配置
    if config:
        save_audio_config(project_path, config)
    
    # 生成 audio_combined.wav
    try:
        regenerate_combined_audio(project_path, config)
        use_original_count = sum(1 for v in config.values() if v) if config else 0
        total_count = len(config) if config else 0
        
        result = tts_result + f"\n\n✅ audio_combined.wav 生成成功！\n"
        result += f"项目: {os.path.basename(project_path)}\n"
        result += f"使用原声的片段: {use_original_count}/{total_count}\n"
        result += f"使用AI配音的片段: {total_count - use_original_count}/{total_count}"
        
        return result
    except Exception as e:
        import traceback
        return tts_result + f"\n\n❌ audio_combined.wav 生成失败: {str(e)}\n{traceback.format_exc()}"

def refresh_projects_for_tts(folder):
    """刷新项目列表（用于TTS界面）"""
    projects = get_all_projects(folder)
    project_choices = [f"{p['name']} - {p['title'][:50]}" for p in projects]
    project_paths = {f"{p['name']} - {p['title'][:50]}": p['path'] for p in projects}
    return gr.Dropdown(choices=project_choices, value=project_choices[0] if project_choices else None), project_paths

def load_segments_from_project(project_choice, project_paths_dict):
    """从选定的项目加载片段列表（从translation.json）"""
    if not project_choice or project_choice not in project_paths_dict:
        return gr.Dataframe(value=[]), "", "请先选择项目"
    
    project_path = project_paths_dict[project_choice]
    
    segments = get_audio_segments(project_path)
    if not segments:
        return gr.Dataframe(value=[]), "", "项目中没有找到音频片段（请确保已生成translation.json）"
    
    # 加载已有配置
    config = load_audio_config(project_path)
    
    # 构建表格数据（只显示编号、翻译文本、使用原声）
    table_data = []
    for segment in segments:
        segment_id = segment['id']
        has_original = segment['has_original']
        use_original = config.get(segment_id, False) if has_original else False
        
        table_data.append([
            segment_id,
            segment['translation'],
            use_original
        ])
    
    project_info = f"项目路径: {project_path}\n音频片段数: {len(segments)}"
    
    return gr.Dataframe(value=table_data, headers=["编号", "翻译文本", "使用原声"], interactive=True), project_path, project_info

# 使用 Blocks 来支持动态交互
with gr.Blocks(title='YouDub - 语音合成') as tts_interafce:
    gr.Markdown("# 语音合成")
    gr.Markdown("一键生成TTS语音合成：配置参数后点击按钮，将自动生成TTS音频并合成 audio_combined.wav")
    
    # TTS生成配置
    tts_folder = gr.Textbox(label='Folder', value='videos', info='视频文件夹路径')
    tts_interface_tts_model = gr.Radio(['bytedance', 'aliyun', 'xtts_v1', 'xtts_v2', 'xtts_v2.0.2'], label='TTS Model', value='xtts_v2',
                                       info='选择语音合成模型：字节跳动、阿里云或 XTTS')
    tts_interface_voice_dropdown = gr.Dropdown(choices=voice_options, label='阿里云音色', value=default_voice_value, visible=False,
                                               info='选择阿里云 CosyVoice 音色（仅当 TTS Model 为 aliyun 时有效）')
    
    # 音频片段配置（在生成前可预览和配置）
    gr.Markdown("### 音频片段配置（可选）")
    gr.Markdown("在生成TTS前，请先选择项目，然后预览并配置哪些片段使用原声。如果不配置，将默认全部使用AI配音。")
    
    # 项目选择
    tts_refresh_projects_btn = gr.Button('刷新项目列表', variant='secondary', size='sm')
    tts_project_dropdown = gr.Dropdown(
        label='选择项目',
        choices=[],
        value=None,
        info='请先选择要配置的项目（需要包含translation.json）'
    )
    tts_project_paths_dict_state = gr.State(value={})  # 存储项目路径字典
    tts_project_path_state = gr.State(value="")  # 存储当前选中的项目路径
    
    # 加载片段列表按钮
    tts_load_segments_btn = gr.Button('加载片段列表', variant='primary', size='sm')
    tts_project_info = gr.Textbox(label='项目信息', lines=3, interactive=False, visible=True)
    
    tts_segments_table = gr.Dataframe(
        headers=["编号", "翻译文本", "使用原声"],
        label='音频片段列表（勾选"使用原声"列来选择使用原声的片段，默认全部AI配音）',
        interactive=True,
        wrap=True,
        datatype=["str", "str", "bool"],
        value=[]
    )
    
    # 当folder改变时自动刷新项目列表
    tts_folder.change(
        fn=refresh_projects_for_tts,
        inputs=[tts_folder],
        outputs=[tts_project_dropdown, tts_project_paths_dict_state]
    )
    
    # 刷新项目列表（保留按钮以备手动刷新）
    tts_refresh_projects_btn.click(
        fn=refresh_projects_for_tts,
        inputs=[tts_folder],
        outputs=[tts_project_dropdown, tts_project_paths_dict_state]
    )
    
    # 加载片段列表
    tts_load_segments_btn.click(
        fn=load_segments_from_project,
        inputs=[tts_project_dropdown, tts_project_paths_dict_state],
        outputs=[tts_segments_table, tts_project_path_state, tts_project_info]
    )
    
    # 当项目选择改变时自动加载
    tts_project_dropdown.change(
        fn=load_segments_from_project,
        inputs=[tts_project_dropdown, tts_project_paths_dict_state],
        outputs=[tts_segments_table, tts_project_path_state, tts_project_info]
    )
    
    gr.Markdown("---")
    
    # 一键生成按钮
    tts_output = gr.Textbox(label='生成结果', lines=10)
    tts_submit_btn = gr.Button('一键生成TTS语音合成', variant='primary', size='lg')
    tts_submit_btn.click(
        fn=generate_tts_with_config,
        inputs=[tts_folder, tts_interface_tts_model, tts_interface_voice_dropdown, tts_segments_table, tts_project_path_state],
        outputs=tts_output
    )
    
    # 添加事件监听（音色下拉框显示/隐藏）
    tts_interface_tts_model.change(
        fn=update_voice_dropdown,
        inputs=[tts_interface_tts_model],
        outputs=[tts_interface_voice_dropdown]
    )

genearte_info_interface = gr.Interface(
    fn = generate_all_info_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),
    ],
    outputs='text',
    flagging_mode='never',
    )

# 视频合成界面（简单版本）
syntehsize_video_interface = gr.Interface(
    fn=synthesize_all_video_under_folder,
    inputs=[
        gr.Textbox(label='Folder', value='videos'),
        gr.Checkbox(label='添加字幕', value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label='速度倍率', value=1.05),
        gr.Slider(minimum=1, maximum=60, step=1, label='帧率', value=30),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], 
                label='分辨率', value='1080p')
    ],
    outputs='text',
    flagging_mode='never',
    title='视频合成'
)

genearte_info_interface = gr.Interface(
    fn = generate_all_info_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),  # Changed 'default' to 'value'
    ],
    outputs='text',
    flagging_mode='never',
)

upload_bilibili_interface = gr.Interface(
    fn = upload_all_videos_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),  # Changed 'default' to 'value'
    ],
    outputs='text',
    flagging_mode='never',
)

app = gr.TabbedInterface(
    interface_list=[do_everything_interface,youtube_interface, demucs_interface,
                    whisper_inference, translation_interface, tts_interafce, syntehsize_video_interface, upload_bilibili_interface],
    tab_names=['全自动', '下载视频', '人声分离', '语音识别', '字幕翻译', '语音合成', '视频合成', '上传B站'],
    title='YouDub')
if __name__ == '__main__':
    app.launch(server_name='0.0.0.0', server_port=7860)
