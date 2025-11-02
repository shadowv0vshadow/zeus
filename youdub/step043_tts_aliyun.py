# coding=utf-8
"""
阿里云百练语音合成 TTS 模块
使用阿里云百练平台（DashScope）的语音合成服务进行文本转语音
支持模型：CosyVoice-v3-plus, CosyVoice-v3, CosyVoice-v2, CosyVoice-v1
参考文档：https://help.aliyun.com/zh/model-studio/cosyvoice-python-sdk
"""

import os
import time
import librosa
import soundfile as sf
import numpy as np
from loguru import logger
from dotenv import load_dotenv
from .utils import save_wav

# 尝试导入 DashScope SDK
DASHSCOPE_AVAILABLE = False
USE_TTS_V2 = False

try:
    import dashscope

    DASHSCOPE_AVAILABLE = True
    # 尝试不同的导入方式
    try:
        # 尝试导入 tts_v2（新版本 API）
        from dashscope.audio.tts_v2 import SpeechSynthesizer as TTSV2Synth

        USE_TTS_V2 = True
    except ImportError:
        try:
            # 尝试导入 tts（旧版本 API）
            from dashscope.audio.tts import SpeechSynthesizer as TTSSynth

            USE_TTS_V2 = False
        except ImportError:
            # 如果都失败，标记为可用但需要运行时检测
            USE_TTS_V2 = False
except ImportError:
    DASHSCOPE_AVAILABLE = False

load_dotenv()

# 阿里云百练（DashScope）配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # 使用与翻译相同的密钥

# 如果 DashScope SDK 可用，设置 API 密钥
if DASHSCOPE_AVAILABLE and DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY


def init_aliyun_tts():
    """初始化阿里云百练 TTS"""
    if not DASHSCOPE_AVAILABLE:
        logger.warning("DashScope SDK 未安装，无法使用阿里云百练 TTS")
        logger.info("提示：如果已安装 dashscope，可能是导入方式问题，代码会自动尝试不同的导入方式")
        return False
    if not DASHSCOPE_API_KEY:
        logger.warning("DASHSCOPE_API_KEY 未设置，无法使用阿里云百练 TTS")
        return False
    dashscope.api_key = DASHSCOPE_API_KEY
    logger.info(f'阿里云百练 TTS 初始化成功，使用模块: {"tts_v2" if USE_TTS_V2 else "tts"}')
    return True


# CosyVoice 支持的音色列表
# 参考：阿里云百炼 CosyVoice 音色列表
COSYVOICE_VOICES = {
    "zh": {
        # 电话销售
        "longyingxiao": "清甜推销女",
        # 短视频配音
        "longjiqi": "呆萌机器人",
        "longhouge": "经典猴哥",
        "longjixin": "毒舌心机女",
        "longanyue": "欢脱粤语男",
        "longgangmei": "TVB港剧国语女",
        "longshange": "原味陕北男",
        "longanmin": "甜美闽南女",
        "longdaiyu": "娇率才女音",
        "longgaoseng": "得道高僧音",
        # 语音助手
        "longanli": "利落从容女",
        "longanlang": "清爽利落男",
        "longanwen": "优雅知性女",
        "longanyun": "居家暖男",
        "longyumi_v2": "正经青年女",
        "longxiaochun_v2": "知性积极女",
        "longxiaoxia_v2": "沉稳权威女",
        # 有声书
        "longyichen": "洒脱活力男",
        "longwanjun": "细腻柔声女",
        "longlaobo": "沧桑岁月爷",
        "longlaoyi": "烟火从容阿姨",
        "longbaizhi": "睿气旁白女",
        "longsanshu": "沉稳质感男",
        "longxiu_v2": "博才说书男",
        "longmiao_v2": "抑扬顿挫女",
        "longyue_v2": "温暖磁性女",
        "longnan_v2": "睿智青年男",
        "longyuan_v2": "温暖治愈女",
        # 社交陪伴
        "longanqin": "亲和活泼女",
        "longanya": "高雅气质女",
        "longanshuo": "干净清爽男",
        "longanling": "思维灵动女",
        "longanzhi": "睿智轻熟男",
        "longanrou": "温柔闺蜜女",
        "longqiang_v2": "浪漫风情女",
        "longhan_v2": "温暖痴情男",
        "longxing_v2": "温婉邻家女",
        "longhua_v2": "元气甜美女",
        "longwan_v2": "积极知性女",
        "longcheng_v2": "智慧青年男",
        "longfeifei_v2": "甜美娇气女",
        "longxiaocheng_v2": "磁性低音男",
        "longzhe_v2": "呆板大暖男",
        "longyan_v2": "温暖春风女",
        "longtian_v2": "磁性理智男",
        "longze_v2": "温暖元气男",
        "longshao_v2": "积极向上男",
        "longhao_v2": "多情忧郁男",
        "kabuleshen_v2": "实力歌手男",
        # 客服
        "longyingmu": "优雅知性女",
        "longyingxun": "年轻青涩男",
        "longyingcui": "严肃催收男",
        "longyingda": "开朗高音女",
        "longyingjing": "低调冷静女",
        "longyingyan": "义正严辞女",
        "longyingtian": "温柔甜美女",
        "longyingbing": "尖锐强势女",
        "longyingtao": "温柔淡定女",
        "longyingling": "温和共情女",
        # 直播带货
        "longanran": "活泼质感女",
        "longanxuan": "经典直播女",
        "longanchong": "激情推销男",
        "longanping": "高亢直播女",
        # 童声
        "longhuhu": "天真烂漫女童",
        "longanpei": "青少年教师女",
        "longwangwang": "台湾少年音",
        "longpaopao": "飞天泡泡音",
        "longshanshan": "戏剧化童声",
        "longniuniu": "阳光男童声",
        "longjielidou_v2": "阳光顽皮男",
        "longling_v2": "稚气呆板女",
        "longke_v2": "懵懂乖乖女",
        "longxian_v2": "豪放可爱女",
        # 方言
        "longlaotie_v2": "东北直率男",
        "longjiayi_v2": "知性粤语女",
        "longtao_v2": "积极粤语女",
        # 诗词朗诵
        "longfei_v2": "热血磁性男",
        "libai_v2": "古代诗仙男",
        "longjin_v2": "优雅温润男",
        # 新闻播报
        "longshu_v2": "沉稳青年男",
        "loongbella_v2": "精准干练女",
        "longshuo_v2": "博才干练男",
        "longxiaobai_v2": "沉稳播报女",
        "longjing_v2": "典型播音女",
        "loongstella_v2": "飒爽利落女",
        # 默认推荐
        "longyue_v2": "温暖磁性女（推荐）",
    },
    "en": {
        "loongeva_v2": "知性英文女（英式）",
        "loongbrian_v2": "沉稳英文男（英式）",
        "loongluna_v2": "英式英文女",
        "loongluca_v2": "英式英文男",
        "loongemily_v2": "英式英文女",
        "loongeric_v2": "英式英文男",
        "loongabby_v2": "美式英文女",
        "loongannie_v2": "美式英文女",
        "loongandy_v2": "美式英文男",
        "loongava_v2": "美式英文女",
        "loongbeth_v2": "美式英文女",
        "loongbetty_v2": "美式英文女",
        "loongcindy_v2": "美式英文女",
        "loongcally_v2": "美式英文女",
        "loongdavid_v2": "美式英文男",
        "loongdonna_v2": "美式英文女",
        # 默认推荐
        "loongabby_v2": "美式英文女（推荐）",
    },
    "ja": {
        "loongyuuna_v2": "元气霓虹女",
        "loongyuuma_v2": "干练霓虹男",
        "loongtomoka_v2": "日语女",
        "loongtomoya_v2": "日语男",
    },
    "ko": {
        "loongjihun_v2": "阳光韩国男",
        "loongkyong_v2": "韩语女",
    },
}

# 默认音色（根据语言自动选择）
DEFAULT_VOICES = {
    "zh": "longyue_v2",  # 温暖磁性女（推荐）
    "en": "loongabby_v2",  # 美式英文女（推荐）
    "ja": "loongyuuna_v2",  # 元气霓虹女
    "ko": "loongjihun_v2",  # 阳光韩国男
}


def tts(text, output_path, speaker_wav=None, model="cosyvoice-v2", language="zh", voice=None):
    """使用阿里云百练 TTS 生成语音

    Args:
        text: 要合成的文本
        output_path: 输出音频文件路径
        speaker_wav: 说话人参考音频（用于声音克隆，CosyVoice-v2 支持，需要音频文件 URL）
        model: 模型名称，可选：
            - 'cosyvoice-v3-plus': CosyVoice-v3-plus（最佳效果，2元/万字符）
            - 'cosyvoice-v3': CosyVoice-v3（平衡质量与价格，0.4元/万字符）
            - 'cosyvoice-v2': CosyVoice-v2（2元/万字符，支持声音复刻）
            - 'cosyvoice-v1': CosyVoice-v1（兼容场景）
        language: 语言代码，'zh' 表示中文，'en' 表示英文，'ja' 表示日语，'ko' 表示韩语
        voice: 音色名称，如 'longyue_v2', 'longyingbing', 'loongabby_v2' 等
            如果不指定，会根据 language 自动选择合适的默认音色
            可用音色列表见 COSYVOICE_VOICES

    Raises:
        ValueError: 当阿里云配置缺失时抛出
    """
    if os.path.exists(output_path):
        logger.info(f"阿里云百练TTS {text} 已存在")
        return

    if not DASHSCOPE_AVAILABLE:
        error_msg = (
            "DashScope SDK 未安装。请运行: pip install dashscope\n" "或者切换到其他 TTS 模型（bytedance 或 xtts_v2）"
        )
        logger.error(error_msg)
        raise ValueError("DashScope SDK 未安装")

    if not DASHSCOPE_API_KEY:
        error_msg = (
            "阿里云百练 TTS 配置缺失。请在 .env 文件中设置：\n"
            "  DASHSCOPE_API_KEY=your_api_key\n\n"
            "该密钥与翻译模块使用的密钥相同（从 bailian.console.aliyun.com 获取）\n"
            "或者切换到其他 TTS 模型（bytedance 或 xtts_v2）"
        )
        logger.error(error_msg)
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

    # 设置 API 密钥
    dashscope.api_key = DASHSCOPE_API_KEY

    for retry in range(3):
        try:
            # CosyVoice 系列模型可以直接用于语音合成，无需声音克隆
            # 参考文档：https://help.aliyun.com/zh/model-studio/cosyvoice-python-sdk
            # 如果需要声音克隆，需要使用 VoiceEnrollmentService 先注册声音（需要音频文件 URL）
            if speaker_wav and os.path.exists(speaker_wav):
                logger.info(f"检测到参考音频: {speaker_wav}，但 CosyVoice 直接生成语音不需要参考音频")
                logger.info("如需声音克隆，请先将音频文件上传到可访问的 URL，然后使用 VoiceEnrollmentService 注册声音")

            # 使用 CosyVoice 模型直接生成语音
            logger.info(f"使用 {model} 进行语音合成")

            if USE_TTS_V2:
                from dashscope.audio.tts_v2 import SpeechSynthesizer

                # 对于 tts_v2，SpeechSynthesizer 初始化时需要 model 和 voice 参数
                # 如果没有指定 voice，根据 language 自动选择默认音色
                if voice is None:
                    voice = DEFAULT_VOICES.get(language, "longyue_v2")
                    logger.info(f"未指定音色，自动选择 {language} 语言的默认音色: {voice}")
                else:
                    # 验证音色是否可用
                    available_voices = COSYVOICE_VOICES.get(language, {})
                    if voice not in available_voices and language == "zh":
                        # 中文音色可能在所有场景中，检查所有中文音色
                        all_zh_voices = {
                            k: v for lang_voices in COSYVOICE_VOICES.values() for k, v in lang_voices.items()
                        }
                        if voice not in all_zh_voices:
                            logger.warning(f"音色 {voice} 可能不存在，尝试使用")
                    logger.info(f"使用指定音色: {voice}")

                synth = SpeechSynthesizer(model=model, voice=voice)
                response = synth.call(text=text)
            else:
                from dashscope.audio.tts import SpeechSynthesizer

                synth = SpeechSynthesizer()
                response = synth.call(model=model, text=text, format="wav", sample_rate=24000)

            # 处理响应
            if response is None:
                raise Exception("TTS API 调用返回 None，请检查 API 配置和网络连接")

            # 检查响应状态
            status_code = getattr(response, "status_code", None)
            if status_code is not None and status_code != 200:
                error_msg = f"TTS API 调用失败: status_code={status_code}, message={getattr(response, 'message', '') or getattr(response, 'code', '未知错误')}"
                logger.warning(f"{error_msg} (重试 {retry + 1}/3)")
                if retry < 2:
                    time.sleep(0.5)
                    continue
                else:
                    raise Exception(error_msg)

            # 提取音频数据
            output = None
            if hasattr(response, "output"):
                output = response.output
            elif status_code is None:
                # 如果没有 status_code，response 可能就是输出
                output = response
            else:
                raise Exception("TTS API 响应格式异常，无法获取输出数据")

            # 尝试获取音频数据
            audio_data = None
            if hasattr(output, "audio"):
                audio_data = output.audio
            elif isinstance(output, dict):
                audio_data = output.get("audio")
            elif isinstance(output, bytes):
                audio_data = output

            if audio_data is None:
                error_msg = f'无法从响应中提取音频数据，output 类型: {type(output)}, 属性: {dir(output)[:10] if hasattr(output, "__dict__") else "N/A"}'
                logger.warning(f"{error_msg} (重试 {retry + 1}/3)")
                if retry < 2:
                    time.sleep(0.5)
                    continue
                else:
                    raise ValueError(error_msg)

            # 保存音频文件（先保存到临时文件）
            import tempfile

            temp_path = output_path + ".tmp"

            if isinstance(audio_data, bytes):
                with open(temp_path, "wb") as f:
                    f.write(audio_data)
            elif isinstance(audio_data, str):
                # 如果是 base64 编码的字符串，需要解码
                import base64

                audio_bytes = base64.b64decode(audio_data)
                with open(temp_path, "wb") as f:
                    f.write(audio_bytes)
            else:
                error_msg = f"无法识别的音频数据格式: {type(audio_data)}"
                logger.warning(f"{error_msg} (重试 {retry + 1}/3)")
                if retry < 2:
                    time.sleep(0.5)
                    continue
                else:
                    raise ValueError(error_msg)

            # 验证并转换音频文件为标准 WAV 格式
            try:
                # 使用 librosa 加载音频（支持多种格式）
                wav, sample_rate = librosa.load(temp_path, sr=24000)

                # 使用 soundfile 保存为标准 WAV 格式（RIFF）
                sf.write(output_path, wav, sample_rate, format="WAV")

                # 删除临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                logger.info(f"阿里云百练TTS 成功: {text[:50]}... -> {output_path} (长度: {len(wav)/sample_rate:.2f}s)")
            except Exception as e:
                # 如果转换失败，尝试直接使用临时文件
                if os.path.exists(temp_path):
                    import shutil

                    shutil.move(temp_path, output_path)
                    logger.warning(f"音频格式转换失败，使用原始文件: {e}")
                else:
                    raise Exception(f"音频验证和转换失败: {e}")

            break  # 成功处理，跳出重试循环

        except Exception as e:
            logger.warning(f"阿里云百练 TTS 失败 (重试 {retry + 1}/3): {e}")
            if retry < 2:
                time.sleep(0.5)
            else:
                raise


# 可用的阿里云百练 TTS 模型列表
# 参考文档：https://help.aliyun.com/zh/model-studio/cosyvoice-python-sdk
ALIYUN_TTS_MODELS = [
    "cosyvoice-v3-plus",  # CosyVoice-v3-plus（最佳效果，2元/万字符）
    "cosyvoice-v3",  # CosyVoice-v3（平衡质量与价格，0.4元/万字符）
    "cosyvoice-v2",  # CosyVoice-v2（2元/万字符，支持声音复刻）
    "cosyvoice-v1",  # CosyVoice-v1（兼容场景）
]


def get_available_models():
    """获取可用的阿里云百练 TTS 模型列表"""
    return ALIYUN_TTS_MODELS


def get_available_voices(language="zh"):
    """获取指定语言可用的音色列表

    Args:
        language: 语言代码，'zh', 'en', 'ja', 'ko'

    Returns:
        dict: 音色名称到描述的映射
    """
    return COSYVOICE_VOICES.get(language, {})


def get_default_voice(language="zh"):
    """获取指定语言的默认音色

    Args:
        language: 语言代码，'zh', 'en', 'ja', 'ko'

    Returns:
        str: 默认音色名称
    """
    return DEFAULT_VOICES.get(language, "longyue_v2")


if __name__ == "__main__":
    # 测试
    test_text = "你好，这是阿里云百练语音合成测试"
    test_output = "test_aliyun_tts.wav"
    test_speaker = None  # 可选：提供参考音频路径
    tts(test_text, test_output, speaker_wav=test_speaker, model="cosyvoice-v2")
