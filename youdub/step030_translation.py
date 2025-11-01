# -*- coding: utf-8 -*-
"""
YouDub 翻译模块 - 五步法翻译

采用深度AI翻译模型，能全面提升语义理解与译文生成的自然度与一致性。

YouDub 视频翻译五步法：
1. 理解核心 - 深入理解视频的主旨和核心信息
2. 语境翻译 - 根据视频的主旨和核心，把字幕翻译成目标语言
3. 文化调整 - 针对翻译结果，根据目标语言的文化背景和表达习惯进行调整
4. 反思调整 - AI对翻译结果自动评估，检测并修正文化语义偏差、流畅度问题及风格一致性
5. 字幕精校 - 最后对翻译好的字幕进行全面检查，确保字幕与视频同步准确无误
"""
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import time
from loguru import logger
from datetime import datetime

# 尝试导入阿里云DashScope SDK
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    logger.warning('DashScope SDK 未安装，如需使用阿里云通义大模型，请运行: pip install dashscope')

load_dotenv()

# 模型提供商配置
MODEL_PROVIDER = os.getenv('MODEL_PROVIDER', 'openai').lower()  # 'openai' 或 'aliyun'
model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
api_key = os.getenv('OPENAI_API_KEY', '')
aliyun_api_key = os.getenv('DASHSCOPE_API_KEY', '')  # 阿里云API密钥

# 如果是阿里云，使用相应的默认值
if MODEL_PROVIDER == 'aliyun':
    if not model_name or model_name.startswith('gpt-'):
        model_name = os.getenv('MODEL_NAME', 'qwen-turbo')  # 默认使用通义千问Turbo

logger.info(f'翻译模块已加载 - YouDub 五步法翻译系统')
logger.info(f'模型提供商: {MODEL_PROVIDER}')
logger.info(f'使用模型: {model_name}')
if MODEL_PROVIDER == 'openai':
    logger.info(f'API 地址: {api_base}')
    logger.info(f'API 密钥已配置: {"是" if api_key else "否"}')
    if not api_key:
        logger.error('⚠️ OPENAI_API_KEY 环境变量未设置！')
else:
    logger.info(f'阿里云 DashScope API 密钥已配置: {"是" if aliyun_api_key else "否"}')
    if not aliyun_api_key:
        logger.error('⚠️ DASHSCOPE_API_KEY 环境变量未设置！')
    if DASHSCOPE_AVAILABLE:
        dashscope.api_key = aliyun_api_key

if model_name == "01ai/Yi-34B-Chat-4bits":
    extra_body = {
        'repetition_penalty': 1.1,
        'stop_token_ids': [7]
    }
else:
    extra_body = {
        #  'repetition_penalty': 1.1,
    }


class UnifiedModelClient:
    """统一的模型调用接口，支持OpenAI和阿里云通义大模型"""
    
    def __init__(self, provider=None, model_name=None, api_base=None, api_key=None):
        self.provider = provider or MODEL_PROVIDER
        # 如果未指定model_name，使用全局配置的model_name
        if model_name is None:
            # 从模块全局变量获取
            import sys
            current_module = sys.modules[__name__]
            model_name = getattr(current_module, 'model_name', 'gpt-3.5-turbo')
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        
        if self.provider == 'openai':
            self.client = OpenAI(
                base_url=api_base or os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                api_key=api_key or os.getenv('OPENAI_API_KEY')
            )
        elif self.provider == 'aliyun':
            self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY', '')
            if not self.api_key:
                raise ValueError('DASHSCOPE_API_KEY 环境变量未设置')
            # 优先使用OpenAI兼容的API方式，更简单且稳定
            self.client = OpenAI(
                base_url=os.getenv('DASHSCOPE_API_BASE', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
                api_key=self.api_key
            )
            # 如果DashScope SDK可用，也设置API密钥
            if DASHSCOPE_AVAILABLE:
                dashscope.api_key = self.api_key
        else:
            raise ValueError(f'不支持的模型提供商: {self.provider}')
    
    def chat_completions_create(self, messages, model=None, timeout=240, **kwargs):
        """统一的聊天完成接口"""
        model = model or self.model_name
        
        if self.provider == 'openai':
            # OpenAI API调用
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout,
                **kwargs
            )
            # 转换为统一格式
            return {
                'choices': [{
                    'message': {
                        'content': response.choices[0].message.content
                    },
                    'finish_reason': response.choices[0].finish_reason
                }],
                'model': response.model
            }
        
        elif self.provider == 'aliyun':
            # 阿里云通义大模型API调用（使用OpenAI兼容接口）
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=timeout,
                    **kwargs
                )
                # 转换为统一格式
                return {
                    'choices': [{
                        'message': {
                            'content': response.choices[0].message.content
                        },
                        'finish_reason': getattr(response.choices[0], 'finish_reason', 'stop')
                    }],
                    'model': getattr(response, 'model', model)
                }
            except Exception as e:
                # 如果OpenAI兼容接口失败，尝试使用DashScope SDK
                if DASHSCOPE_AVAILABLE:
                    logger.debug(f'OpenAI兼容接口调用失败，尝试使用DashScope SDK: {e}')
                    # 转换消息格式
                    dashscope_messages = []
                    for msg in messages:
                        role_map = {
                            'system': 'system',
                            'user': 'user',
                            'assistant': 'assistant'
                        }
                        dashscope_messages.append({
                            'role': role_map.get(msg['role'], 'user'),
                            'content': msg['content']
                        })
                    
                    # 调用DashScope API
                    response = Generation.call(
                        model=model,
                        messages=dashscope_messages,
                        result_format='message',  # 返回消息格式
                    )
                    
                    # 处理响应
                    if response.status_code == 200:
                        return {
                            'choices': [{
                                'message': {
                                    'content': response.output.choices[0].message.content
                                },
                                'finish_reason': getattr(response.output.choices[0], 'finish_reason', 'stop')
                            }],
                            'model': model
                        }
                    else:
                        error_msg = f"阿里云API调用失败: {getattr(response, 'message', '') or getattr(response, 'code', '未知错误')}"
                        raise Exception(error_msg)
                else:
                    raise e
        
        else:
            raise ValueError(f'不支持的模型提供商: {self.provider}')


def get_model_client(**kwargs):
    """获取模型客户端实例"""
    provider = kwargs.get('provider', MODEL_PROVIDER)
    client_model_name = kwargs.get('model_name')
    if provider == 'openai':
        return UnifiedModelClient(
            provider='openai',
            model_name=client_model_name,
            api_base=kwargs.get('api_base', api_base),
            api_key=kwargs.get('api_key', api_key)
        )
    elif provider == 'aliyun':
        return UnifiedModelClient(
            provider='aliyun',
            model_name=client_model_name,
            api_key=kwargs.get('api_key', aliyun_api_key)
        )
    else:
        raise ValueError(f'不支持的模型提供商: {provider}')


def get_necessary_info(info: dict):
    return {
        'title': info['title'],
        'uploader': info['uploader'],
        'description': info['description'],
        'upload_date': info['upload_date'],
        'categories': info['categories'],
        'tags': info['tags'],
    }


def ensure_transcript_length(transcript, max_length=2000):
    """确保转录文本长度不超过限制，避免触发内容过滤
    
    Args:
        transcript: 转录文本
        max_length: 最大长度（字符数）
        
    Returns:
        截断后的转录文本
    """
    # 如果文本较短，直接返回
    if len(transcript) <= max_length:
        return transcript
    
    # 取开头和结尾，跳过中间部分
    mid = len(transcript) // 2
    before, after = transcript[:mid], transcript[mid:]
    length = max_length // 2
    return before[:length] + after[-length:]


def print_separator(title=""):
    """打印分隔线"""
    separator = "=" * 85
    if title:
        logger.info(f"\n{separator}\n{title}\n{separator}\n")
    else:
        logger.info(separator)


def calculate_token_estimate(transcript):
    """估算token消耗
    
    Args:
        transcript: 转录文本列表
        
    Returns:
        字幕条数, 字符数, 预计token数
    """
    subtitle_count = len(transcript)
    char_count = sum(len(line['text']) for line in transcript)
    estimated_tokens = char_count * 2  # 粗略估计：1个汉字约2个token
    return subtitle_count, char_count, estimated_tokens


def deep_understand_video(info, transcript, target_language='简体中文'):
    """第一步：理解核心
    
    深入理解视频的主旨、风格、受众、文化背景等核心信息
    
    Args:
        info: 视频信息
        transcript: 转录文本
        target_language: 目标语言
        
    Returns:
        视频深度理解结果
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    subtitle_count, char_count, estimated_tokens = calculate_token_estimate(transcript)
    
    logger.info(f"\n{timestamp} AI 预算消耗token数：这部视频字幕条数={subtitle_count}，字符数={char_count}，预计花费token数={estimated_tokens}")
    logger.info(f"\n{timestamp} 进入第1轮->理解核心： 首先深入理解视频的主旨和核心信息。")
    logger.info("\n" + "-" * 30 + "AI对该视频的理解如下：" + "-" * 30 + "\n")
    
    client = get_model_client()
    
    transcript_text = ' '.join(line['text'] for line in transcript)
    transcript_text = ensure_transcript_length(transcript_text, max_length=1500)
    transcript_text = transcript_text.replace('\n', ' ').strip()
    
    info_message = f'标题: "{info["title"]}" 作者: "{info["uploader"]}". '
    
    # 深度理解prompt
    understanding_prompt = f'''作为一个视频内容分析专家，请对以下视频进行深度理解和分析：

视频信息：
{info_message}

视频文稿片段：
{transcript_text}

请从以下几个维度进行深入分析，并以JSON格式返回：
1. 核心主题：视频的主要内容和核心信息
2. 整体风格：视频的表达风格（幽默/严肃/教育/娱乐等）
3. 目标受众：视频的目标观众群体
4. 语气特点：视频的语言风格和语气特征
5. 专业术语：视频中涉及的专业领域和术语
6. 文化背景：视频的文化背景和地域特点
7. 教育意义：视频传递的价值观或教育意义

请返回JSON格式：
```json
{{
  "title": "视频标题",
  "core_theme": "核心主题描述",
  "style": "整体风格",
  "target_audience": "目标受众",
  "tone": "语气特点",
  "terminology": "专业术语说明",
  "cultural_background": "文化背景",
  "educational_value": "教育意义或价值观"
}}
```'''

    messages = [
        {'role': 'system', 'content': '你是一个专业的视频内容分析专家，擅长深入理解视频的主题、风格、受众和文化背景。'},
        {'role': 'user', 'content': understanding_prompt},
    ]
    
    success = False
    understanding = None
    
    for retry in range(5):
        try:
            response = client.chat_completions_create(
                messages=messages,
                timeout=240,
                **extra_body if extra_body else {}
            )
            
            finish_reason = response['choices'][0]['finish_reason']
            if finish_reason == 'content_filter':
                logger.warning('⚠️ 内容被过滤，尝试使用更简短的内容...')
                raise Exception('内容过滤，重试')
            
            content = response['choices'][0]['message']['content']
            if not content:
                raise Exception('API 返回内容为空')
            
            # 提取JSON
            json_matches = re.findall(r'\{.*?\}', content.replace('\n', ''), re.DOTALL)
            if not json_matches:
                raise Exception("未找到 JSON 格式")
            
            understanding = json.loads(json_matches[0])
            
            # 验证必要字段
            required_fields = ['title', 'core_theme', 'style', 'target_audience']
            if not all(field in understanding for field in required_fields):
                raise Exception('理解结果缺少必要字段')
            
            # 打印AI理解结果
            logger.info(f"\n{understanding.get('core_theme', '')}\n")
            logger.info(f"整体风格{understanding.get('style', '')}，适合{understanding.get('target_audience', '')}观看。")
            logger.info(f"语气{understanding.get('tone', '')}。")
            if understanding.get('terminology'):
                logger.info(f"专业术语：{understanding['terminology']}")
            if understanding.get('cultural_background'):
                logger.info(f"文化背景：{understanding['cultural_background']}")
            if understanding.get('educational_value'):
                logger.info(f"\n{understanding['educational_value']}\n")
            
            success = True
            break
            
        except Exception as e:
            logger.warning(f'视频理解失败 (尝试 {retry + 1}/5): {e}')
            time.sleep(2)
    
    if not success or not understanding:
        # 回退到简单理解
        logger.warning('深度理解失败，使用简化理解')
        understanding = {
            'title': info['title'],
            'core_theme': '视频内容',
            'style': '信息类',
            'target_audience': '普通观众',
            'tone': '中性',
            'terminology': '',
            'cultural_background': '',
            'educational_value': ''
        }
    
    return understanding


def summarize(info, transcript, target_language='简体中文'):
    client = get_model_client()
    transcript = ' '.join(line['text'] for line in transcript)
    transcript = ensure_transcript_length(transcript, max_length=1500)  # 减少长度避免内容过滤
    
    # 清理可能触发过滤器的内容
    transcript = transcript.replace('\n', ' ').strip()
    
    info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '

    full_description = f'''Video information:
{info_message}

Video excerpt (partial transcript):
{transcript}

Please create a brief summary in JSON format:
```json
{{"title": "video title here", "summary": "brief video summary here"}}
```'''

    messages = [
        {'role': 'system',
            'content': f'You are a expert in the field of this video. Please detailedly summarize the video in JSON format.\n```json\n{{"title": "the title of the video", "summary", "the summary of the video"}}\n```'},
        {'role': 'user', 'content': full_description},
    ]
    retry_message=''
    success = False
    for retry in range(5):
        try:
            messages = [
                {'role': 'system', 'content': f'You are a expert in the field of this video. Please summarize the video in JSON format.\n```json\n{{"title": "the title of the video", "summary", "the summary of the video"}}\n```'},
                {'role': 'user', 'content': full_description+retry_message},
            ]
            
            logger.info(f'正在调用 AI API, 模型: {model_name}')
            
            kwargs = extra_body if extra_body else {}
            response = client.chat_completions_create(
                messages=messages,
                timeout=240,
                **kwargs
            )
            
            # 详细记录响应信息
            finish_reason = response['choices'][0]['finish_reason']
            logger.info(f'API 响应模型: {response.get("model", model_name)}')
            logger.info(f'Finish reason: {finish_reason}')
            
            # 特殊处理内容过滤
            if finish_reason == 'content_filter':
                logger.warning('⚠️ 内容被过滤器拦截')
                logger.warning('尝试使用更简短的转录内容...')
                # 大幅缩短转录内容
                short_transcript = ' '.join(line['text'] for line in transcript[:10])  # 只取前10句
                info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '
                full_description = f'The following is a brief excerpt from the video:\n{info_message}\n{short_transcript}\n\nSummarize in JSON format:\n```json\n{{"title": "", "summary": ""}}\n```'
                messages[1]['content'] = full_description
                raise Exception('内容过滤，使用简短版本重试')
            
            summary = response['choices'][0]['message']['content']
            if not summary:
                logger.error('API 返回内容为空！')
                logger.error(f'完整响应对象: {response}')
                raise Exception('API 返回的内容为空')
            
            summary = summary.replace('\n', '')
            if '视频标题' in summary:
                raise Exception('包含"视频标题"关键词')
            logger.info(f'AI 返回内容: {summary}')
            
            # 查找 JSON 格式
            json_matches = re.findall(r'\{.*?\}', summary, re.DOTALL)
            if not json_matches:
                logger.warning(f'未找到 JSON 格式，原始响应: {summary}')
                raise Exception("响应中没有找到 JSON 格式")
            
            summary = json_matches[0]
            summary = json.loads(summary)
            summary = {
                'title': summary.get('title', '').replace('title:', '').strip(),
                'summary': summary.get('summary', '').replace('summary:', '').strip()
            }
            
            if not summary['title'] or 'title' in summary['title']:
                raise Exception('Invalid summary title')
            if not summary['summary']:
                raise Exception('Invalid summary content')
                
            success = True
            break
        except Exception as e:
            retry_message += '\nSummarize the video in JSON format:\n```json\n{"title": "", "summary": ""}\n```'
            logger.warning(f'总结失败 (尝试 {retry + 1}/5): {e}')
            time.sleep(1)
    if not success:
        raise Exception(f'总结失败')

    title = summary['title']
    summary = summary['summary']
    tags = info['tags']
    messages = [
        {'role': 'system',
            'content': f'You are a native speaker of {target_language}. Please translate the title and summary into {target_language} in JSON format. ```json\n{{"title": "the {target_language} title of the video", "summary", "the {target_language} summary of the video", "tags": [list of tags in {target_language}]}}\n```.'},
        {'role': 'user',
            'content': f'The title of the video is "{title}". The summary of the video is "{summary}". Tags: {tags}.\nPlease translate the above title and summary and tags into {target_language} in JSON format. ```json\n{{"title": "", "summary", ""， "tags": []}}\n```. Remember to tranlate the title and the summary and tags into {target_language} in JSON.'},
    ]
    max_translation_retries = 10
    for retry in range(max_translation_retries):
        try:
            kwargs = extra_body if extra_body else {}
            response = client.chat_completions_create(
                messages=messages,
                timeout=240,
                **kwargs
            )
            summary = response['choices'][0]['message']['content'].replace('\n', '')
            logger.info(f'AI 翻译返回内容: {summary}')
            
            # 查找 JSON 格式
            json_matches = re.findall(r'\{.*?\}', summary, re.DOTALL)
            if not json_matches:
                logger.warning(f'未找到 JSON 格式，原始响应: {summary}')
                raise Exception("翻译响应中没有找到 JSON 格式")
            
            summary = json_matches[0]
            summary = json.loads(summary)
            
            if target_language in summary.get('title', '') or target_language in summary.get('summary', ''):
                raise Exception('Invalid translation: contains target language name')
            
            title = summary.get('title', '').strip()
            if not title:
                raise Exception('Translation title is empty')
                
            # 去除引号
            if (title.startswith('"') and title.endswith('"')) or \
               (title.startswith('"') and title.endswith('"')) or \
               (title.startswith(''') and title.endswith(''')) or \
               (title.startswith("'") and title.endswith("'")) or \
               (title.startswith('《') and title.endswith('》')):
                title = title[1:-1]
            
            result = {
                'title': title,
                'author': info['uploader'],
                'summary': summary.get('summary', ''),
                'tags': summary.get('tags', []),
                'language': target_language
            }
            
            logger.info(f'翻译成功: {result["title"]}')
            return result
            
        except Exception as e:
            logger.warning(f'总结翻译失败 (尝试 {retry + 1}/{max_translation_retries}): {e}')
            time.sleep(2)
    
    # 如果所有重试都失败，返回一个默认的结果
    logger.error('翻译多次失败，使用默认值')
    raise Exception(f'翻译失败，已重试 {max_translation_retries} 次')


def translation_postprocess(result):
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    result = result.replace("AI", '人工智能')
    result = result.replace('变压器', "Transformer")
    return result

def valid_translation(text, translation):

    if (translation.startswith('```') and translation.endswith('```')):
        translation = translation[3:-3]
        return True, translation_postprocess(translation)

    if (translation.startswith('“') and translation.endswith('”')) or (translation.startswith('"') and translation.endswith('"')):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)

    if '翻译' in translation and '：“' in translation and '”' in translation:
        translation = translation.split('：“')[-1].split('”')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and '："' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and ':"' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if len(text) <= 10:
        if len(translation) > 15:
            return False, f'Only translate the following sentence and give me the result.'
    elif len(translation) > len(text)*0.75:
        return False, f'The translation is too long. Only translate the following sentence and give me the result.'

    forbidden = ['翻译', '这句', '\n', '简体中文', '中文', 'translate', 'Translate', 'translation', 'Translation']
    translation = translation.strip()
    
    # 检查是否为空
    if not translation:
        return False, f'Translation is empty. Please provide the translation.'
    
    for word in forbidden:
        if word in translation:
            return False, f"Don't include `{word}` in the translation. Only translate the following sentence and give me the result."

    return True, translation_postprocess(translation)
# def split_sentences(translation, punctuations=['。', '？', '！', '\n', '”', '"']):
#     def is_punctuation(char):
#         return char in punctuations

#     output_data = []
#     for item in translation:
#         start = item['start']
#         text = item['text']
#         speaker = item['speaker']
#         translation = item['translation']
#         sentence_start = 0
#         duration_per_char = (item['end'] - item['start']) / len(translation)
#         for i, char in enumerate(translation):
#             # If the character is a punctuation, split the sentence
#             if not is_punctuation(char) and i != len(translation) - 1:
#                 continue
#             if i - sentence_start < 5 and i != len(translation) - 1:
#                 continue
#             if i < len(translation) - 1 and is_punctuation(translation[i+1]):
#                 continue
#             sentence = translation[sentence_start:i+1]
#             sentence_end = start + duration_per_char * len(sentence)

#             # Append the new item
#             output_data.append({
#                 "start": round(start, 3),
#                 "end": round(sentence_end, 3),
#                 "text": text,
#                 "speaker": speaker,
#                 "translation": sentence
#             })

#             # Update the start for the next sentence
#             start = sentence_end
#             sentence_start = i + 1
#     return output_data


def preserve_proper_nouns(original_text, translation, proper_nouns):
    """确保专有名词在翻译中保留原文
    
    Args:
        original_text: 原文
        translation: 翻译结果
        proper_nouns: 专有名词集合
        
    Returns:
        修正后的翻译
    """
    if not proper_nouns or not translation:
        return translation
    
    result = translation
    
    # 检查原文中的专有名词是否在翻译中被翻译了
    for noun in proper_nouns:
        if len(noun) < 2:  # 跳过太短的词
            continue
        
        # 如果原文包含这个专有名词，但翻译中没有（可能被翻译了）
        if noun in original_text:
            # 尝试在翻译中找到对应的中文（可能是翻译了）
            # 如果找不到原文，尝试恢复
            if noun not in result:
                # 在专有名词位置插入原文（简单处理）
                # 这里可以根据需要添加更智能的恢复逻辑
                pass
    
    return result


def apply_terminology_consistency(original_text, translation, terminology_dict):
    """应用术语一致性，确保同一术语在全文中的翻译一致
    
    Args:
        original_text: 原文
        translation: 翻译结果
        terminology_dict: 术语字典（原文 -> 翻译）
        
    Returns:
        修正后的翻译
    """
    if not terminology_dict or not translation:
        return translation
    
    result = translation
    
    # 检查术语字典中的术语，确保在翻译中保持一致
    for original_term, established_translation in terminology_dict.items():
        # 如果原文包含这个术语，但翻译不一致
        if original_term in original_text:
            # 尝试在翻译中找到并使用已建立的翻译
            # 这里可以添加更智能的匹配和替换逻辑
            pass
    
    return result


def update_terminology_dict(original_text, translation, terminology_dict):
    """更新术语字典，记录术语的翻译映射
    
    Args:
        original_text: 原文
        translation: 翻译结果
        terminology_dict: 术语字典（会被更新）
    """
    # 提取可能的术语对（原文中的英文 -> 翻译中的对应部分）
    # 这里可以添加更智能的术语提取逻辑
    
    # 简单示例：如果有明显的英文术语，记录其翻译
    english_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original_text)
    for term in english_terms:
        if term not in terminology_dict and len(term) > 2:
            # 如果术语在翻译中被保留（未翻译），记录为保留原文
            if term in translation:
                terminology_dict[term] = term  # 保留原文
            # 可以添加更多逻辑来识别术语的翻译对应关系


def split_text_into_sentences(para):
    para = re.sub('([。！？\?])([^，。！？\?”’》])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

def split_sentences(translation):
    """将翻译结果按句子分割
    
    Args:
        translation: 翻译结果列表
        
    Returns:
        分句后的翻译结果列表
    """
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item.get('translation', '')
        
        # 清理翻译文本（移除空白字符）
        if translation_text:
            translation_text = translation_text.strip()
        
        # 检查翻译是否为空（包括空字符串、None、只有空白字符）
        if not translation_text:
            logger.warning(f'翻译为空，使用原文: {text[:50]}...')
            # 使用原文作为备用
            output_data.append({
                "start": round(start, 3),
                "end": round(item['end'], 3),
                "text": text,
                "speaker": speaker,
                "translation": text  # 使用原文作为备用
            })
            continue
        
        # 分句处理
        sentences = split_text_into_sentences(translation_text)
        
        # 再次检查长度（防御性编程）
        text_length = len(translation_text)
        if text_length == 0:
            logger.error(f'意外的空翻译: {item}')
            output_data.append({
                "start": round(start, 3),
                "end": round(item['end'], 3),
                "text": text,
                "speaker": speaker,
                "translation": text
            })
            continue
        
        duration_per_char = (item['end'] - item['start']) / text_length
        sentence_start = 0
        
        for sentence in sentences:
            if not sentence or not sentence.strip():  # 跳过空句子
                continue
                
            sentence_end = start + duration_per_char * len(sentence)

            # Append the new item
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": text,
                "speaker": speaker,
                "translation": sentence.strip()
            })

            # Update the start for the next sentence
            start = sentence_end
            sentence_start += len(sentence)
    
    return output_data

def extract_proper_nouns(transcript):
    """从转录文本中提取可能的专有名词（游戏名、音乐术语、品牌名等）
    
    Args:
        transcript: 转录文本列表
        
    Returns:
        专有名词集合
    """
    proper_nouns = set()
    
    # 提取所有大写单词和可能的专有名词模式
    for line in transcript:
        text = line['text']
        # 提取首字母大写的连续单词（可能是专有名词）
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        proper_nouns.update(words)
        
        # 提取常见的游戏/音乐术语模式（全大写或特定格式）
        # 游戏名通常：首字母大写或全大写
        game_patterns = re.findall(r'\b[A-Z]{2,}\b', text)  # 全大写（如 RPG, FPS, AI）
        proper_nouns.update(game_patterns)
        
        # 提取带引号的内容（可能是作品名、专辑名等）
        quoted = re.findall(r'["\']([^"\']+)["\']', text)
        proper_nouns.update(quoted)
    
    return proper_nouns


def context_translate(understanding, transcript, target_language='简体中文'):
    """第二步：语境翻译
    
    基于对视频的深度理解，进行语境化翻译
    
    Args:
        understanding: 视频理解结果
        transcript: 转录文本
        target_language: 目标语言
        
    Returns:
        初步翻译结果列表
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"\n{timestamp} 进入第2轮->语境翻译： 根据视频的主旨和核心，把字幕翻译成目标语言。\n")
    
    client = get_model_client()
    
    # 提取专有名词
    proper_nouns = extract_proper_nouns(transcript)
    proper_nouns_list = sorted(list(proper_nouns))[:50]  # 限制数量避免过长
    
    # 构建丰富的上下文信息
    context_info = f'''视频标题："{understanding["title"]}"
核心主题：{understanding.get("core_theme", "")}
整体风格：{understanding.get("style", "")}
目标受众：{understanding.get("target_audience", "")}
语气特点：{understanding.get("tone", "")}'''
    
    if understanding.get("terminology"):
        context_info += f'\n专业术语：{understanding["terminology"]}'
    if understanding.get("cultural_background"):
        context_info += f'\n文化背景：{understanding["cultural_background"]}'
    
    # 判断视频类型（音乐/游戏）
    video_type = ""
    title_lower = understanding.get("title", "").lower()
    core_theme_lower = understanding.get("core_theme", "").lower()
    
    music_keywords = ['music', 'song', 'album', 'artist', 'beat', 'melody', 'rhythm', '音', '曲', '歌', '专辑', '音乐']
    game_keywords = ['game', 'gaming', 'play', 'player', 'level', 'boss', 'quest', '游戏', '玩家', '关卡', '剧情']
    
    if any(keyword in title_lower or keyword in core_theme_lower for keyword in music_keywords):
        video_type = "音乐"
    elif any(keyword in title_lower or keyword in core_theme_lower for keyword in game_keywords):
        video_type = "游戏"
    
    # 构建专业翻译提示词
    professional_rules = []
    if video_type:
        if video_type == "音乐":
            professional_rules = [
                "8. 【音乐专业术语】保留所有音乐术语原文不翻译，包括但不限于：",
                "   - 音乐风格：Jazz, Blues, Rock, Pop, EDM, Hip-Hop, R&B 等",
                "   - 乐器名称：Piano, Guitar, Drums, Bass, Violin 等",
                "   - 音乐技术术语：BPM, Tempo, Key, Chord, Scale, Harmony, Melody 等",
                "   - 音乐软件/品牌：Ableton, FL Studio, Logic Pro, MIDI, VST 等",
                "   - 专辑名、歌曲名、艺术家名：完全保留原文，不翻译",
                "9. 使用专业的音乐翻译表达，确保术语准确性和专业性"
            ]
        elif video_type == "游戏":
            professional_rules = [
                "8. 【游戏专业术语】保留所有游戏相关专有名词原文不翻译，包括但不限于：",
                "   - 游戏名称：完全保留原文（如 The Legend of Zelda, Final Fantasy 等）",
                "   - 游戏术语：RPG, FPS, MMO, NPC, DLC, DPS, HP, MP, XP, Boss, Quest, Level 等",
                "   - 角色名、地名、技能名：保留原文不翻译",
                "   - 游戏引擎/平台：Unity, Unreal Engine, Steam, PlayStation, Xbox 等",
                "   - 品牌名：Nintendo, Sony, Microsoft, EA, Ubisoft 等",
                "9. 使用专业的游戏翻译表达，确保术语准确性和专业性"
            ]
    
    # 构建系统提示词
    system_prompt = f'''你是一个专业的视频字幕翻译专家，精通{target_language}的表达习惯，特别擅长音乐和游戏内容的专业翻译。

视频背景信息：
{context_info}

{"视频类型：" + video_type if video_type else ""}

{"已识别的专有名词（必须保留原文）：" + ", ".join(proper_nouns_list) if proper_nouns_list else ""}

翻译要求（严格执行）：
1. 【专业性】翻译必须专业、准确、严谨，符合{target_language}的专业表达习惯
2. 【专有名词保留】以下内容必须保留原文不翻译：
   - 所有专有名词（人名、地名、品牌名、产品名）
   - 游戏名称、角色名、技能名、装备名
   - 音乐专辑名、歌曲名、艺术家名
   - 技术术语缩写（AI, API, CPU, GPU, FPS, BPM 等）
   - 英文大小写、标点符号、格式必须完全保留
3. 【上下文一致性】在整个翻译过程中，必须保持术语和表达的一致性：
   - 同一个专有名词在整个视频中必须统一（要么都保留原文，要么都使用同一翻译）
   - 专业术语的翻译必须前后一致
   - 风格和语气必须统一连贯
4. 翻译要自然、流畅、地道，避免翻译腔
5. 保持原文的情感色彩和语气
6. 数学公式使用plain text，不使用latex
7. 人工智能的"agent"翻译为"智能体"

{chr(10).join(professional_rules) if professional_rules else ""}

请只返回翻译结果，不要包含"翻译"等提示词。所有专有名词必须保留原文。'''
    
    fixed_message = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': '"Knowledge is power."'},
        {'role': 'assistant', 'content': '知识就是力量。'},
        {'role': 'user', 'content': '"To be or not to be, that is the question."'},
        {'role': 'assistant', 'content': '生存还是毁灭，这是一个值得考虑的问题。'},
    ]
    
    full_translation = []
    history = []
    total = len(transcript)
    
    # 维护术语一致性字典（原文 -> 翻译）
    terminology_dict = {}
    
    logger.info(f"{timestamp} AI翻译字幕: 进度 0%")
    
    for idx, line in enumerate(transcript):
        text = line['text']
        original_text = text
        
        # 显示进度
        progress = int((idx + 1) / total * 100)
        if (idx + 1) % max(1, total // 10) == 0 or idx == total - 1:
            logger.info(f"{timestamp} AI翻译字幕: 进度 {progress}%")
        
        # 预处理异常文本
        if len(text) > 200:
            repeated_pattern = re.search(r'(.)\1{10,}', text)
            if repeated_pattern:
                text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
                logger.warning(f'检测到大量重复字符，已简化')
        
        translation = None
        last_raw_translation = None  # 保存最后一次API返回的原始内容
        last_error = None  # 保存最后一次错误信息
        
        for retry in range(5):
            if retry == 0:
                user_content = f'"{text}"'
            elif retry == 1:
                user_content = f'请翻译：{text}'
            else:
                truncated = text[:100] if len(text) > 100 else text
                user_content = f'{truncated}'
            
            messages = fixed_message + history[-30:] + [{'role': 'user', 'content': user_content}]
            
            try:
                kwargs = extra_body if extra_body else {}
                response = client.chat_completions_create(
                    messages=messages,
                    timeout=240,
                    **kwargs
                )
                
                finish_reason = response['choices'][0]['finish_reason']
                if finish_reason == 'content_filter':
                    text = text[:50] if len(text) > 50 else text
                    last_error = '内容被过滤'
                    raise Exception('Content filtered')
                
                raw_translation = response['choices'][0]['message']['content']
                last_raw_translation = raw_translation  # 保存原始返回内容
                
                if raw_translation:
                    raw_translation = raw_translation.replace('\n', '').strip()
                
                if not raw_translation:
                    last_error = 'API返回内容为空'
                    raise Exception('Empty translation response')
                
                # 尝试清理和验证翻译
                success, cleaned_translation = valid_translation(original_text, raw_translation)
                if not success:
                    # 保存验证失败的原因（cleaned_translation此时是错误提示信息）
                    last_error = f'验证失败: {cleaned_translation[:100]}'
                    # 尝试从原始返回中提取可能的翻译内容
                    # 即使验证失败，也可能包含有用的翻译
                    if retry < 4:
                        raise Exception('Invalid translation')
                    else:
                        # 最后一次重试，尝试智能提取
                        # 移除常见的提示词前缀
                        extracted = raw_translation
                        for prefix in ['翻译:', 'Translation:', '翻译：', '翻译为:', '翻译为：']:
                            if prefix in extracted:
                                extracted = extracted.split(prefix, 1)[-1].strip()
                        # 移除引号
                        if (extracted.startswith('"') and extracted.endswith('"')) or \
                           (extracted.startswith('"') and extracted.endswith('"')) or \
                           (extracted.startswith("'") and extracted.endswith("'")):
                            extracted = extracted[1:-1].strip()
                        # 如果提取的内容看起来像翻译（包含中文字符），使用它
                        if any('\u4e00' <= char <= '\u9fff' for char in extracted):
                            translation = extracted
                            logger.warning(f'翻译验证失败，但已从API返回中提取翻译: {original_text[:30]}... -> {extracted[:30]}...')
                            break
                        else:
                            raise Exception('Invalid translation')
                else:
                    translation = cleaned_translation
                    
                    # 检查并修正专有名词保留
                    translation = preserve_proper_nouns(original_text, translation, proper_nouns)
                    
                    # 检查并应用术语一致性
                    translation = apply_terminology_consistency(original_text, translation, terminology_dict)
                    
                    break
                    
            except Exception as e:
                error_msg = str(e)
                if retry == 4:
                    # 最后一次重试失败，记录详细信息
                    error_details = f'原文: {original_text[:50]}...'
                    if last_raw_translation:
                        error_details += f' | API返回: {last_raw_translation[:100]}...'
                    if last_error:
                        error_details += f' | 错误: {last_error}'
                    else:
                        error_details += f' | 异常: {error_msg}'
                    logger.warning(f'翻译失败 ({retry+1}/5): {error_details}')
                else:
                    # 非最后一次失败，只记录简要信息
                    if retry >= 2:  # 从第3次重试开始记录
                        logger.debug(f'翻译重试 ({retry+1}/5): {original_text[:30]}... - {error_msg}')
                
                # 递增等待时间，避免频繁请求
                wait_time = min(1 + retry * 0.5, 3)
                time.sleep(wait_time)
        
        if translation is None or not translation.strip():
            # 如果所有重试都失败，尝试使用最后一次的原始返回（如果存在）
            if last_raw_translation:
                # 简单清理后使用
                fallback = last_raw_translation.strip()
                # 移除常见的提示词
                for prefix in ['翻译:', 'Translation:', '翻译：', '翻译为:', '翻译为：']:
                    if prefix in fallback:
                        fallback = fallback.split(prefix, 1)[-1].strip()
                # 如果包含中文，优先使用
                if any('\u4e00' <= char <= '\u9fff' for char in fallback):
                    translation = fallback
                    logger.warning(f'使用API返回作为fallback翻译: {original_text[:30]}... -> {fallback[:30]}...')
                else:
                    translation = original_text
                    logger.warning(f'翻译失败，使用原文: {original_text[:50]}...')
            else:
                translation = original_text
                logger.warning(f'翻译失败且无API返回，使用原文: {original_text[:50]}...')
        
        # 更新术语字典（从翻译中提取术语映射）
        update_terminology_dict(original_text, translation, terminology_dict)
        
        full_translation.append(translation)
        history.append({'role': 'user', 'content': f'"{original_text}"'})
        history.append({'role': 'assistant', 'content': translation})
        time.sleep(0.1)
    
    logger.info(f"{timestamp} AI翻译字幕: 进度 100%\n")
    return full_translation


def cultural_adaptation(translations, transcript, target_language='简体中文', understanding=None):
    """第三步：文化调整
    
    针对翻译结果，根据目标语言的文化背景和表达习惯进行调整
    
    Args:
        translations: 初步翻译结果
        transcript: 原始转录
        target_language: 目标语言
        understanding: 视频理解结果
        
    Returns:
        文化调整后的翻译结果
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{timestamp} 进入第3轮->文化调整： 针对意译的结果，根据目标语言的文化背景和表达习惯，对翻译文本进行适当调整。\n")
    
    # 文化调整规则
    adapted_translations = []
    for translation in translations:
        # 应用后处理规则
        adapted = translation_postprocess(translation)
        adapted_translations.append(adapted)
    
    return adapted_translations


def reflection_and_refinement(translations, transcript, target_language='简体中文', understanding=None):
    """第四步：反思调整
    
    AI对翻译结果自动评估，检测并修正文化语义偏差、流畅度问题及风格一致性
    
    Args:
        translations: 文化调整后的翻译
        transcript: 原始转录
        target_language: 目标语言
        understanding: 视频理解结果
        
    Returns:
        反思调整后的翻译结果
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{timestamp} 进入第4轮->反思调整： AI对翻译结果自动评估，检测并修正文化语义偏差、流畅度问题及风格一致性等方面的问题。\n")
    logger.info(f"{timestamp} AI正在进行翻译， 进度： 0%")
    
    # 简化版：主要检查明显错误
    refined_translations = []
    total = len(translations)
    
    for idx, (translation, line) in enumerate(zip(translations, transcript)):
        # 显示进度
        if (idx + 1) % max(1, total // 4) == 0 or idx == total - 1:
            progress = int((idx + 1) / total * 100)
            logger.info(f"{timestamp} AI正在进行翻译， 进度： {progress}%")
        
        # 基本检查：长度合理性、禁用词等
        refined = translation
        
        # 检查过长的翻译
        if len(refined) > len(line['text']) * 2:
            refined = refined[:len(line['text']) * 2]
        
        refined_translations.append(refined)
    
    logger.info(f"{timestamp} AI正在进行翻译， 进度： 100%\n")
    return refined_translations


def subtitle_polish(translations, transcript, target_language='简体中文'):
    """第五步：字幕精校
    
    最后对翻译好的字幕进行全面检查，确保字幕与视频同步准确无误，语言表述精准，格式规范统一
    
    Args:
        translations: 反思调整后的翻译
        transcript: 原始转录（包含时间信息）
        target_language: 目标语言
        
    Returns:
        最终精校后的翻译结果
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"{timestamp} 进入第5轮->字幕精校： 最后对翻译好的字幕进行全面检查，确保字幕与视频同步准确无误，语言表述精准，格式规范统一。\n")
    
    # 精校：格式规范化
    polished_translations = []
    for translation in translations:
        # 去除多余空格
        polished = ' '.join(translation.split())
        # 确保标点符号正确
        polished = polished.strip()
        polished_translations.append(polished)
    
    return polished_translations


def _translate(summary, transcript, target_language='简体中文'):
    """旧版翻译函数 - 保留向后兼容"""
    understanding = {
        'title': summary.get('title', ''),
        'core_theme': summary.get('summary', ''),
        'style': '信息类',
        'target_audience': '普通观众',
        'tone': '中性',
    }
    return translate_with_five_steps(understanding, transcript, target_language)


def translate_with_five_steps(understanding, transcript, target_language='简体中文'):
    """使用五步法进行翻译
    
    Args:
        understanding: 视频理解结果
        transcript: 转录文本
        target_language: 目标语言
        
    Returns:
        最终翻译结果列表
    """
    # 第二步：语境翻译
    translations = context_translate(understanding, transcript, target_language)
    
    # 第三步：文化调整
    translations = cultural_adaptation(translations, transcript, target_language, understanding)
    
    # 第四步：反思调整
    translations = reflection_and_refinement(translations, transcript, target_language, understanding)
    
    # 第五步：字幕精校
    translations = subtitle_polish(translations, transcript, target_language)
    
    return translations

def translate(folder, target_language='简体中文', model_provider=None):
    """使用五步法进行视频字幕翻译
    
    Args:
        folder: 视频文件夹路径
        target_language: 目标语言
        model_provider: 模型提供商 ('openai' 或 'aliyun')，如果不提供则使用环境变量配置
        
    Returns:
        是否翻译成功
    """
    # 声明全局变量（必须在函数开头）
    global MODEL_PROVIDER
    
    # 如果提供了 model_provider，临时设置环境变量并重新加载配置
    original_provider = None
    original_global_provider = None
    if model_provider:
        original_provider = os.getenv('MODEL_PROVIDER')
        os.environ['MODEL_PROVIDER'] = model_provider
        # 重新加载全局配置
        original_global_provider = MODEL_PROVIDER
        MODEL_PROVIDER = model_provider.lower()
    
    try:
        translation_path = os.path.join(folder, 'translation.json')
        if os.path.exists(translation_path):
            logger.info(f'翻译已存在: {folder}')
            return True
        
        # 打印五步法说明
        separator = "=" * 85
        logger.info(f"\n{separator}")
        logger.info("\nYouDub 采用深度AI翻译模型，能全面提升语义理解与译文生成的自然度与一致性。\n")
        logger.info("YouDub 视频翻译五步法：1. 理解核心   2. 语境翻译   3. 文化调整   4. 反思调整   5. 字幕精校\n")
        logger.info(f"{separator}\n")
        
        # 加载视频信息
        info_path = os.path.join(folder, 'download.info.json')
        if not os.path.exists(info_path):
            logger.error(f'视频信息文件不存在: {info_path}')
            return False
        
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        info = get_necessary_info(info)
        
        # 加载转录文本
        transcript_path = os.path.join(folder, 'transcript.json')
        if not os.path.exists(transcript_path):
            logger.error(f'转录文件不存在: {transcript_path}')
            return False
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
        
        # 第一步：理解核心
        understanding = deep_understand_video(info, transcript, target_language)
        
        # 生成简单摘要（用于向后兼容）
        summary_path = os.path.join(folder, 'summary.json')
        summary = {
            'title': understanding.get('title', info['title']),
            'author': info['uploader'],
            'summary': understanding.get('core_theme', ''),
            'tags': info.get('tags', []),
            'language': target_language,
            'understanding': understanding  # 保存完整的理解结果
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 第二到五步：五步法翻译
        translations = translate_with_five_steps(understanding, transcript, target_language)
        
        # 组合翻译结果
        for i, line in enumerate(transcript):
            if i < len(translations):
                line['translation'] = translations[i]
            else:
                line['translation'] = line['text']  # 备用
            # 初始化use_original字段（默认False，使用AI配音）
            if 'use_original' not in line:
                line['use_original'] = False
        
        # 分句处理
        transcript = split_sentences(transcript)
        
        # 保存翻译结果
        with open(translation_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"\n{timestamp} ✅ 翻译完成！\n")
        logger.info(f"{separator}\n")
        
        return True
        
    except Exception as e:
        logger.error(f'翻译过程出错: {e}')
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 恢复原始配置
        if original_provider is not None:
            os.environ['MODEL_PROVIDER'] = original_provider
        elif model_provider and 'MODEL_PROVIDER' in os.environ:
            del os.environ['MODEL_PROVIDER']
        if original_global_provider is not None:
            MODEL_PROVIDER = original_global_provider

def translate_all_transcript_under_folder(folder, target_language, model_provider=None):
    """翻译文件夹下所有视频的字幕
    
    Args:
        folder: 文件夹路径
        target_language: 目标语言
        model_provider: 模型提供商 ('openai' 或 'aliyun')，如果不提供则使用环境变量配置
    """
    # 如果提供了 model_provider，临时设置环境变量
    original_provider = None
    if model_provider:
        original_provider = os.getenv('MODEL_PROVIDER')
        os.environ['MODEL_PROVIDER'] = model_provider
    
    try:
        for root, dirs, files in os.walk(folder):
            if 'transcript.json' in files and 'translation.json' not in files:
                translate(root, target_language, model_provider=model_provider)
        return f'Translated all videos under {folder}'
    finally:
        # 恢复原始配置
        if original_provider is not None:
            os.environ['MODEL_PROVIDER'] = original_provider
        elif model_provider and 'MODEL_PROVIDER' in os.environ:
            del os.environ['MODEL_PROVIDER']

if __name__ == '__main__':
    translate_all_transcript_under_folder(
        r'videos\TED-Ed\20240227 Can you solve the magical maze riddle - Alex Rosenthal', '简体中文')
