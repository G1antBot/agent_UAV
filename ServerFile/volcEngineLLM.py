import requests
import re
from runtime_logger import get_runtime_logger


class VolcEngineFakeHFModel:
    def __init__(self, api_key=None, api_url=None, model_id=None):
        # 初始化API密钥、API地址和模型ID
        self.api_key = api_key or "24572520-5c64-4470-8c3d-5ecb84781725"  # 火山引擎API密钥
        self.api_url = api_url or "https://ark.cn-beijing.volces.com/api/v3/chat/completions"  # 火山引擎API地址
        self.model_id = model_id or "deepseek-v3-250324"  # 使用的模型ID
        self.logger = get_runtime_logger("llm")

    def _normalize_code(self, text: str) -> str:
        """
        统一提取可执行Python代码，兼容<code>与```python两种包裹格式。
        """
        code = text.strip()

        # 先剥离 <code>...</code>
        code_tag_match = re.search(r"<code>\s*([\s\S]*?)\s*</code>", code, flags=re.IGNORECASE)
        if code_tag_match:
            code = code_tag_match.group(1).strip()

        # 再剥离 ```python ... ```
        fence_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", code, flags=re.IGNORECASE)
        if fence_match:
            code = fence_match.group(1).strip()

        return code

    def generate(self, messages, **kwargs):
        # 将传入的消息对象序列化为字典格式，因为API需要这种格式
        serialized_messages = [{"role": m.role, "content": m.content} for m in messages]

        # 设置请求头，包括内容类型和授权信息
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        # 构造请求负载，包括模型ID、消息序列、温度参数和是否流式传输
        payload = {"model": self.model_id, "messages": serialized_messages, "temperature": 0.7, "stream": False}

        # 定义一个模拟的Token使用情况类，用于返回模拟的token使用信息
        class FakeTokenUsage:
            def __init__(self):
                self.input_tokens = 0  # 输入token数量
                self.output_tokens = 0  # 输出token数量
                self.total_tokens = 0  # 总token数量

        try:
            self.logger.info(f"请求火山方舟模型: model={self.model_id}, messages={len(serialized_messages)}")
            # 发送POST请求到API
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.encoding = 'utf-8'  # 设置响应编码为UTF-8
            response.raise_for_status()  # 如果响应状态码不是200，抛出异常

            # 解析响应内容，获取生成的消息内容
            raw_content = response.json()["choices"][0]["message"]["content"]
            code = self._normalize_code(raw_content)
            self.logger.info("火山方舟响应成功")
            self.logger.debug(f"归一化代码预览: {code[:200]}")

            # 定义一个模拟的消息类，用于返回处理后的消息内容
            class FakeMessage:
                def __init__(self, content):
                    self.content = f"<code>\n{code}\n</code>"  # 将代码内容包装在<code>标签中
                    self.token_usage = FakeTokenUsage()  # 初始化模拟的token使用情况

            print(f"火山方舟 API 调用成功。")  # 打印成功信息
            return FakeMessage(code)  # 返回模拟的消息对象

        except Exception as e:
            # 如果API调用失败，打印错误信息
            print(f"火山方舟 API 调用失败: {e}")
            self.logger.error(f"火山方舟 API 调用失败: {e}")

            # 定义一个模拟的消息类，用于返回失败的响应
            class FakeMessage:
                def __init__(self, content):
                    self.content = ("<code>\nprint('失败')\n</code>")  # 返回一个简单的失败代码
                    self.token_usage = FakeTokenUsage()  # 初始化模拟的token使用情况

            return FakeMessage("【模型响应失败】")  # 返回模拟的消息对象
