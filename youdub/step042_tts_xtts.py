import os
from TTS.api import TTS
from loguru import logger
import numpy as np
import torch
import time
from .utils import save_wav

# 设置 TTS 模型存储目录到项目的 models/TTS 目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tts_models_dir = os.path.join(project_root, 'models', 'TTS')
# 设置 XDG_DATA_HOME 环境变量，让 TTS 库从项目目录查找模型
# Coqui TTS 会在 XDG_DATA_HOME/tts 目录下查找模型（使用 -- 分隔符的目录名）
if 'XDG_DATA_HOME' not in os.environ:
    os.environ['XDG_DATA_HOME'] = tts_models_dir
# 确保目录存在
os.makedirs(tts_models_dir, exist_ok=True)

model = None


def init_TTS():
    load_model()


def load_model(model_path="tts_models/multilingual/multi-dataset/xtts_v2", device='auto'):
    global model
    if model is not None:
        return
    
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Loading TTS model from {model_path}')
    t_start = time.time()

    from torch.serialization import add_safe_globals

    # ====== 导入所有可能被序列化的类 ======
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import (
        XttsAudioConfig,
        XttsArgs,
    )
    from TTS.config.shared_configs import (
        BaseDatasetConfig,
        BaseAudioConfig,
    )
    from TTS.tts.configs.shared_configs import (
        BaseTTSConfig,
        CharactersConfig,
    )

    # ====== 一次性加入所有可信类 ======
    add_safe_globals([
        XttsConfig,
        XttsAudioConfig,
        XttsArgs,
        BaseDatasetConfig,
        BaseAudioConfig,
        BaseTTSConfig,
        CharactersConfig,
    ])

    model = TTS(model_path).to(device)
    t_end = time.time()
    logger.info(f'TTS model loaded in {t_end - t_start:.2f}s')


def tts(text, output_path, speaker_wav, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device='auto', language='zh-cn'):
    global model

    if os.path.exists(output_path):
        logger.info(f'TTS {text} 已存在')
        return

    if model is None:
        load_model(model_name, device)

    for retry in range(3):
        try:
            wav = model.tts(text, speaker_wav=speaker_wav, language=language)
            wav = np.array(wav)
            save_wav(wav, output_path)
            logger.info(f'TTS {text}')
            break
        except Exception as e:
            logger.warning(f'TTS {text} 失败')
            logger.warning(e)


if __name__ == '__main__':
    speaker_wav = r'videos\TED-Ed\20231121 Why did the US try to kill all the bison？ - Andrew C. Isenberg\audio_vocals.wav'
    while True:
        text = input('请输入：')
        tts(text, f'playground/{text}.wav', speaker_wav)
