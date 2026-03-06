"""
配置管理模块
- 管理 API Key 的运行时存储
- 提供全局配置常量
"""
from pydantic import BaseModel
from typing import Optional


class APIKeys(BaseModel):
    """用户提交的 API Key 存储模型"""
    openai_key: Optional[str] = None
    gemini_key: Optional[str] = None


# 全局运行时配置（单实例，非持久化）
api_keys = APIKeys()

# 上传目录
UPLOAD_DIR = "uploads"

# 翻译系统提示词：设定为专业学术翻译助手
TRANSLATION_SYSTEM_PROMPT = """你是一位专业的学术论文翻译助手。请将以下英文学术文本翻译为中文。
要求：
1. 保持学术论文的正式语体
2. 专业术语使用学界通用的中文翻译，首次出现时在括号内保留英文原文
3. 数学公式、变量名、算法名称保持英文原文不翻译
4. 保持段落结构不变，不要合并或拆分段落
5. 图表标题和参考文献编号保持原样"""

# 分析系统提示词
ANALYSIS_SYSTEM_PROMPT = """你是一位资深的科研论文分析助手。用户会提供一篇完整的学术论文内容作为上下文。
请基于论文内容回答用户的问题。回答时：
1. 引用论文中的具体内容作为依据
2. 使用清晰的结构化表达
3. 对专业概念进行适当解释
4. 如果问题超出论文范围，请明确说明"""
