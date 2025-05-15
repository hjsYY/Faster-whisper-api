import requests
from requests.exceptions import RequestException
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transcribe_audio(file_path, api_url="http://localhost:8001/transcribe"):
    """
    调用语音识别API

    :param file_path: 音频文件路径
    :param api_url: API端点地址
    :return: 转录结果字典或None
    """
    try:
        # 检查文件是否存在
        with open(file_path, 'rb') as audio_file:
            files = {'file': (file_path, audio_file, 'audio/wav')}

            # 添加超时设置(连接5秒，读取30秒)
            response = requests.post(
                api_url,
                files=files,
                timeout=(5, 30)
            )

            # 检查响应状态
            response.raise_for_status()

            return response.json()

    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
    except RequestException as e:
        logger.error(f"API请求失败: {str(e)}")
    except ValueError as e:
        logger.error(f"JSON解析失败: {str(e)}")
    return None

if __name__ == "__main__":
    result = transcribe_audio("./2.wav")
    if result:
        print("转录结果:")
        print(f"语言: {result.get('language')}")
        print(f"时长: {result.get('duration')}秒")
        print(f"文本: {result.get('text')}")
    else:
        print("转录失败")
