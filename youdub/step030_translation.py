# -*- coding: utf-8 -*-
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
import time
from loguru import logger

load_dotenv()

model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
api_key = os.getenv('OPENAI_API_KEY', '')

logger.info(f'Translation module loaded')
logger.info(f'Using model: {model_name}')
logger.info(f'API base: {api_base}')
logger.info(f'API key configured: {"Yes" if api_key else "No"}')

if not api_key:
    logger.error('⚠️ OPENAI_API_KEY is not set in environment variables!')
if model_name == "01ai/Yi-34B-Chat-4bits":
    extra_body = {
        'repetition_penalty': 1.1,
        'stop_token_ids': [7]
    }
else:
    extra_body = {
        #  'repetition_penalty': 1.1,
    }


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


def summarize(info, transcript, target_language='简体中文'):
    client = OpenAI(
    # This is the default and can be omitted
    base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
    api_key=os.getenv('OPENAI_API_KEY')
)
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
            
            logger.info(f'正在调用 OpenAI API, 模型: {model_name}')
            
            if extra_body:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=240,
                    **extra_body
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=240
                )
            
            # 详细记录响应信息
            finish_reason = response.choices[0].finish_reason
            logger.info(f'API 响应模型: {response.model}')
            logger.info(f'Finish reason: {finish_reason}')
            
            # 特殊处理内容过滤
            if finish_reason == 'content_filter':
                logger.warning('⚠️ 内容被 OpenAI 过滤器拦截')
                logger.warning('尝试使用更简短的转录内容...')
                # 大幅缩短转录内容
                short_transcript = ' '.join(line['text'] for line in transcript[:10])  # 只取前10句
                info_message = f'Title: "{info["title"]}" Author: "{info["uploader"]}". '
                full_description = f'The following is a brief excerpt from the video:\n{info_message}\n{short_transcript}\n\nSummarize in JSON format:\n```json\n{{"title": "", "summary": ""}}\n```'
                messages[1]['content'] = full_description
                raise Exception('内容过滤，使用简短版本重试')
            
            summary = response.choices[0].message.content
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
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                timeout=240,
                extra_body=extra_body
            )
            summary = response.choices[0].message.content.replace('\n', '')
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

def _translate(summary, transcript, target_language='简体中文'):
    client = OpenAI(
        # This is the default and can be omitted
        base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        api_key=os.getenv('OPENAI_API_KEY')
    )
    info = f'This is a video called "{summary["title"]}". {summary["summary"]}.'
    full_translation = []
    fixed_message = [
        {'role': 'system', 'content': f'You are a expert in the field of this video.\n{info}\nTranslate the sentence into {target_language}.下面我让你来充当翻译家，你的目标是把任何语言翻译成中文，请翻译时不要带翻译腔，而是要翻译得自然、流畅和地道，使用优美和高雅的表达方式。请将人工智能的“agent”翻译为“智能体”，强化学习中是`Q-Learning`而不是`Queue Learning`。数学公式写成plain text，不要使用latex。确保翻译正确和简洁。注意信达雅。'},
        {'role': 'user', 'content': '使用地道的中文Translate:"Knowledge is power."'},
        {'role': 'assistant', 'content': '翻译：“知识就是力量。”'},
        {'role': 'user', 'content': '使用地道的中文Translate:"To be or not to be, that is the question."'},
        {'role': 'assistant', 'content': '翻译：“生存还是毁灭，这是一个值得考虑的问题。”'},]

    history = []
    for line in transcript:
        text = line['text']
        original_text = text
        
        # 策略1: 预处理异常文本（大量重复字符）
        # 检测是否有异常长的重复字符序列
        if len(text) > 200:
            # 检查是否有超过10个连续重复的字符
            import re
            repeated_pattern = re.search(r'(.)\1{10,}', text)
            if repeated_pattern:
                # 简化重复字符为最多3个
                text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
                logger.warning(f'检测到大量重复字符，已简化: {original_text[:50]}... -> {text[:50]}...')

        retry_message = 'Only translate the quoted sentence and give me the final translation.'
        translation = None
        
        for retry in range(5):
            # 策略2: 根据重试次数调整策略
            if retry == 0:
                # 第一次：正常翻译
                user_content = f'使用地道的中文Translate:"{text}"'
            elif retry == 1:
                # 第二次：简化提示
                user_content = f'Translate to Chinese: {text}'
            elif retry == 2:
                # 第三次：更直接的指令
                user_content = f'请翻译成中文：{text}'
            else:
                # 第四、五次：截短文本重试
                truncated = text[:100] if len(text) > 100 else text
                user_content = f'翻译：{truncated}'
            
            messages = fixed_message + history[-30:] + [{'role': 'user', 'content': user_content}]

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=240,
                    extra_body=extra_body
                )
                
                # 策略3: 检查 finish_reason
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'content_filter':
                    logger.warning(f'内容被过滤 (content_filter)，尝试简化文本')
                    # 大幅简化文本
                    text = text[:50] if len(text) > 50 else text
                    raise Exception('Content filtered, retry with shorter text')
                
                translation = response.choices[0].message.content
                if translation:
                    translation = translation.replace('\n', '')
                
                logger.info(f'原文：{original_text[:100]}...' if len(original_text) > 100 else f'原文：{original_text}')
                logger.info(f'译文：{translation}')
                
                # 检查是否为空响应
                if not translation or not translation.strip():
                    logger.warning(f'AI 返回空响应，重试 {retry + 1}/5')
                    raise Exception('Empty translation response')
                
                success, translation = valid_translation(original_text, translation)
                if not success:
                    retry_message += translation
                    raise Exception('Invalid translation')
                break
            except Exception as e:
                logger.error(f'翻译失败 (尝试 {retry + 1}/5): {e}')
                if str(e) == 'Internal Server Error':
                    client = OpenAI(
                        base_url=os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1'),
                        api_key=os.getenv('OPENAI_API_KEY')
                    )
                time.sleep(1)
        
        # 策略4: 如果5次重试后仍然失败，使用原文作为后备
        if translation is None or not translation.strip():
            logger.warning(f'翻译失败5次，使用原文作为后备: {original_text[:50]}...')
            translation = original_text
        
        full_translation.append(translation)
        history.append({'role': 'user', 'content': f'Translate:"{original_text}"'})
        history.append({'role': 'assistant', 'content': f'翻译："{translation}"'})
        time.sleep(0.1)

    return full_translation

def translate(folder, target_language='简体中文'):
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True

    info_path = os.path.join(folder, 'download.info.json')
    if not os.path.exists(info_path):
        return False
    # info_path = r'videos\Lex Clips\20231222 Jeff Bezos on fear of death ｜ Lex Fridman Podcast Clips\download.info.json'
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    info = get_necessary_info(info)

    transcript_path = os.path.join(folder, 'transcript.json')
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)

    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path, 'r', encoding='utf-8'))
    else:
        summary = summarize(info, transcript, target_language)
        if summary is None:
            logger.error(f'Failed to summarize {folder}')
            return False
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    translation_path = os.path.join(folder, 'translation.json')
    translation = _translate(summary, transcript, target_language)
    for i, line in enumerate(transcript):
        line['translation'] = translation[i]
    transcript = split_sentences(transcript)
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return True

def translate_all_transcript_under_folder(folder, target_language):
    for root, dirs, files in os.walk(folder):
        if 'transcript.json' in files and 'translation.json' not in files:
            translate(root, target_language)
    return f'Translated all videos under {folder}'

if __name__ == '__main__':
    translate_all_transcript_under_folder(
        r'videos\TED-Ed\20240227 Can you solve the magical maze riddle - Alex Rosenthal', '简体中文')
