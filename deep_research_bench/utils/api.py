import os
from typing import Optional, Dict, Any
from google import genai
from google.genai import types
import requests
import logging
import time  # 引入 time 模块
from functools import wraps # 引入 wraps
import warnings  # 添加 warnings 模块

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


logging.getLogger('google').setLevel(logging.WARNING)
logging.getLogger('google.genai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# 过滤 Google Genai SDK 的 BlockedReason 警告
warnings.filterwarnings('ignore', message='.*is not a valid BlockedReason.*')

# Read API keys from environment variables
API_KEY = os.environ.get("GEMINI_API_KEY", "")
READ_API_KEY = os.environ.get("JINA_API_KEY", "")
FACT_Model = "gemini-2.5-flash-preview-05-20"
Model = "gemini-2.5-pro-preview-06-05"

class AIClient:
    
    def __init__(self, api_key=API_KEY, model=Model):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided! Please set GEMINI_API_KEY environment variable.")
        
        # Configure client
        # self.client = genai.Client(http_options=types.HttpOptions(base_url='https://api.uniapi.io/gemini'), api_key=self.api_key, http_options={'timeout': 600000})
        http_config = types.HttpOptions(
            base_url='https://api.uniapi.io/gemini', 
            timeout=600000
        )

        self.client = genai.Client(api_key=self.api_key, http_options=http_config)
        self.model = model
        
    def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        """
        Generate text response with robust error handling
        """
        model_to_use = model or self.model
        
        # Build request content
        contents = []
        
        # Add system prompt
        if system_prompt:
            contents.append({
                "role": "system",
                "parts": [{"text": system_prompt}]
            })
        
        # Add user prompt
        contents.append({
            "role": "user", 
            "parts": [{"text": user_prompt}]
        })
        
        try:
            response = self.client.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=16000)
                )
            )
            
            # 尝试获取响应文本，处理可能的异常
            try:
                return response.text
            except Exception as text_error:
                # 如果 response.text 失败，尝试其他方式获取内容
                logger.warning(f"Failed to get response.text: {text_error}")
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        parts_text = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                parts_text.append(part.text)
                        if parts_text:
                            return ''.join(parts_text)
                raise text_error
            
        except Exception as e:
            raise Exception(f"Failed to generate content: {str(e)}")

# class WebScrapingJinaTool:
#     def __init__(self, api_key: str = None):
#         self.api_key = api_key or os.environ.get("JINA_API_KEY")
#         if not self.api_key:
#             raise ValueError("Jina API key not provided! Please set JINA_API_KEY environment variable.")

#     def __call__(self, url: str) -> Dict[str, Any]:
#         try:
#             jina_url = f'https://r.jina.ai/{url}'
#             headers = {
#                 "Accept": "application/json",
#                 'Authorization': self.api_key,
#                 'X-Timeout': "60000",
#                 "X-With-Generated-Alt": "true",
#             }
#             response = requests.get(jina_url, headers=headers)

#             if response.status_code != 200:
#                 raise Exception(f"Jina AI Reader Failed for {url}: {response.status_code}")

#             response_dict = response.json()

#             return {
#                 'url': response_dict['data']['url'],
#                 'title': response_dict['data']['title'],
#                 'description': response_dict['data']['description'],
#                 'content': response_dict['data']['content'],
#                 'publish_time': response_dict['data'].get('publishedTime', 'unknown')
#             }

#         except Exception as e:
#             logger.error(str(e))
#             return {
#                 'url': url,
#                 'content': '',
#                 'error': str(e)
#             }

# --- 【核心修改】为 Jina 工具添加重试装饰器 ---
def retry_with_backoff(retries=3, backoff_in_seconds=2):
    """
    一个装饰器，用于在函数调用失败时进行重试，并采用指数退避策略。
    """
    def rwb(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            _retries, _delay = retries, backoff_in_seconds
            while _retries > 1:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    # 仅在遇到5xx系列服务器错误时重试
                    if "524" in str(e) or "502" in str(e) or "503" in str(e) or "504" in str(e):
                        logger.warning(f"请求失败，错误: {e}. 将在 {_delay} 秒后重试...")
                        time.sleep(_delay)
                        _retries -= 1
                        _delay *= 2  # 指数增加延迟
                    else:
                        # 如果是其他错误 (如 4xx 客户端错误)，则不重试，直接抛出
                        raise e
            # 最后一次尝试
            return f(*args, **kwargs)
        return wrapper
    return rwb

class WebScrapingJinaTool:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not provided! Please set JINA_API_KEY environment variable.")

    @retry_with_backoff(retries=3, backoff_in_seconds=3) # <--- 在这里应用重试装饰器
    def __call__(self, url: str) -> Dict[str, Any]:
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                "Accept": "application/json",
                'Authorization': f'Bearer {self.api_key}', # <--- 官方推荐使用 Bearer Token
                'X-Timeout': "60000",
                "X-With-Generated-Alt": "true",
            }
            # 增加一个请求超时参数，避免本地请求永久等待
            response = requests.get(jina_url, headers=headers, timeout=120) 

            # 检查状态码，如果不是 200，主动抛出异常以触发重试
            if response.status_code != 200:
                # 抛出异常，这样我们的重试装饰器就能捕捉到它
                raise Exception(f"Jina AI Reader Failed for {url}: {response.status_code} - {response.text}")

            response_dict = response.json()

            return {
                'url': response_dict['data']['url'],
                'title': response_dict['data']['title'],
                'description': response_dict['data']['description'],
                'content': response_dict['data']['content'],
                'publish_time': response_dict['data'].get('publishedTime', 'unknown')
            }

        except Exception as e:
            # 在这里记录错误信息，然后重新抛出，让装饰器处理
            logger.error(f"在抓取 {url} 时发生无法恢复的错误: {str(e)}")
            # 返回一个标准的错误结构体，而不是重新抛出，这样即使最终失败，程序也不会崩溃
            return {
                'url': url,
                'title': '',
                'description': '',
                'content': '',
                'error': str(e)
            }


jina_tool = WebScrapingJinaTool()

def scrape_url(url: str) -> Dict[str, Any]:
    return jina_tool(url)
    
def call_model(user_prompt: str) -> str:
    client = AIClient(model=FACT_Model)
    return client.generate(user_prompt)

# if __name__ == "__main__":
#     url = ""
#     result = scrape_url(url)
#     print(result)

if __name__ == "__main__":
    # 示例：如何使用修改后的 call_model 函数
    # try:
    #     # 确保您已设置 UNIAPI_KEY 环境变量
    #     prompt = "你好，请介绍一下你自己。"
    #     response_text = call_model(prompt)
    #     print("模型响应:")
    #     print(response_text)
    # except Exception as e:
    #     print(f"发生错误: {e}")
    # 示例1：测试一个可能会超时的 URL
    url_to_test = "https://www.alliedmarketresearch.com/orthodontics-market"
    print(f"正在尝试抓取: {url_to_test}")
    result = scrape_url(url_to_test)
    print("\n抓取结果:")
    print(result)