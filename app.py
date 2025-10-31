import gradio as gr
from youdub.step000_video_downloader import download_from_url
from youdub.step010_demucs_vr import separate_all_audio_under_folder
from youdub.step020_whisperx import transcribe_all_audio_under_folder
from youdub.step030_translation import translate_all_transcript_under_folder
from youdub.step040_tts import generate_all_wavs_under_folder
from youdub.step050_synthesize_video import synthesize_all_video_under_folder
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
    allow_flagging='never',
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
    allow_flagging='never',
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
    allow_flagging='never',
)

translation_interface = gr.Interface(
    fn=translate_all_transcript_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),  # Changed 'default' to 'value'
        gr.Dropdown(['简体中文', '繁体中文', 'English', 'Deutsch', 'Français', 'русский'],
                    label='Target Language', value='简体中文'),
        gr.Radio(['openai', 'aliyun'], label='Model Provider', value='openai',
                 info='选择AI模型提供商：OpenAI 或 阿里云通义大模型'),
    ],
    outputs='text',
)

def tts_interface_wrapper(folder, tts_model, aliyun_voice_str):
    """TTS 接口包装函数，解析音色字符串"""
    aliyun_voice = None
    if tts_model == 'aliyun' and aliyun_voice_str:
        # 从 "voice_id - description" 格式中提取 voice_id
        aliyun_voice = aliyun_voice_str.split(' - ')[0] if ' - ' in aliyun_voice_str else aliyun_voice_str
    return generate_all_wavs_under_folder(folder, tts_model=tts_model, aliyun_voice=aliyun_voice)

# 使用 Blocks 来支持动态交互
with gr.Blocks(title='YouDub - 语音合成') as tts_interafce:
    tts_folder = gr.Textbox(label='Folder', value='videos')
    tts_interface_tts_model = gr.Radio(['bytedance', 'aliyun', 'xtts_v1', 'xtts_v2', 'xtts_v2.0.2'], label='TTS Model', value='xtts_v2',
                                       info='选择语音合成模型：字节跳动、阿里云或 XTTS')
    tts_interface_voice_dropdown = gr.Dropdown(choices=voice_options, label='阿里云音色', value=default_voice_value, visible=False,
                                               info='选择阿里云 CosyVoice 音色（仅当 TTS Model 为 aliyun 时有效）')
    tts_output = gr.Textbox(label='Output', lines=10)
    
    tts_submit_btn = gr.Button('Submit')
    tts_submit_btn.click(
        fn=tts_interface_wrapper,
        inputs=[tts_folder, tts_interface_tts_model, tts_interface_voice_dropdown],
        outputs=tts_output
    )
    
    # 添加事件监听
    tts_interface_tts_model.change(
        fn=update_voice_dropdown,
        inputs=[tts_interface_tts_model],
        outputs=[tts_interface_voice_dropdown]
    )
syntehsize_video_interface = gr.Interface(
    fn=synthesize_all_video_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),  # Changed 'default' to 'value'
        gr.Checkbox(label='Subtitles', value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label='Speed Up', value=1.05),
        gr.Slider(minimum=1, maximum=60, step=1, label='FPS', value=30),
        gr.Radio(['4320p', '2160p', '1440p', '1080p', '720p', '480p', '360p', '240p', '144p'], label='Resolution', value='1080p'),
    ],
    outputs='text',
    allow_flagging='never',
)

genearte_info_interface = gr.Interface(
    fn = generate_all_info_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),  # Changed 'default' to 'value'
    ],
    outputs='text',
    allow_flagging='never',
)

upload_bilibili_interface = gr.Interface(
    fn = upload_all_videos_under_folder,
    inputs = [
        gr.Textbox(label='Folder', value='videos'),  # Changed 'default' to 'value'
    ],
    outputs='text',
    allow_flagging='never',
)

app = gr.TabbedInterface(
    interface_list=[do_everything_interface,youtube_interface, demucs_interface,
                    whisper_inference, translation_interface, tts_interafce, syntehsize_video_interface, upload_bilibili_interface],
    tab_names=['全自动', '下载视频', '人声分离', '语音识别', '字幕翻译', '语音合成', '视频合成', '上传B站'],
    title='YouDub')
if __name__ == '__main__':
    app.launch()
