"""
分析服务模块
=============
调用 Gemini API (gemini-1.5-flash) 进行论文内容的智能问答分析。
支持流式输出 (SSE) 和多轮对话。
"""

import google.generativeai as genai
from typing import AsyncGenerator
from app.config import api_keys, ANALYSIS_SYSTEM_PROMPT


async def analyze_with_context(
    paper_text: str,
    user_question: str,
    chat_history: list[dict] | None = None,
) -> str:
    """非流式分析（用于简短问答）"""
    if not api_keys.gemini_key:
        raise ValueError("请先在配置中心设置 Gemini API Key")

    genai.configure(api_key=api_keys.gemini_key)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=ANALYSIS_SYSTEM_PROMPT,
    )

    context_prompt = (
        f"以下是需要分析的论文全文：\n\n"
        f"---论文开始---\n{paper_text}\n---论文结束---\n\n"
    )

    if chat_history:
        history = [
            {"role": "user", "parts": [context_prompt + "请阅读以上论文，准备好回答我的问题。"]},
            {"role": "model", "parts": ["我已仔细阅读了这篇论文，请随时提问。"]},
        ]
        history.extend(chat_history)
        chat = model.start_chat(history=history)
        response = chat.send_message(user_question)
    else:
        full_prompt = context_prompt + f"用户问题：{user_question}"
        response = model.generate_content(full_prompt)

    return response.text


async def analyze_stream(
    paper_text: str,
    user_question: str,
    chat_history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """
    流式分析 — 逐块返回文本，用于 SSE 打字机效果。

    使用 Gemini 的 stream=True 参数，每生成一个文本块就立即 yield，
    前端通过 EventSource 接收实时输出。
    """
    if not api_keys.gemini_key:
        raise ValueError("请先在配置中心设置 Gemini API Key")

    genai.configure(api_key=api_keys.gemini_key)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=ANALYSIS_SYSTEM_PROMPT,
    )

    context_prompt = (
        f"以下是需要分析的论文全文：\n\n"
        f"---论文开始---\n{paper_text}\n---论文结束---\n\n"
    )

    if chat_history:
        history = [
            {"role": "user", "parts": [context_prompt + "请阅读以上论文，准备好回答我的问题。"]},
            {"role": "model", "parts": ["我已仔细阅读了这篇论文，请随时提问。"]},
        ]
        history.extend(chat_history)
        chat = model.start_chat(history=history)
        response = chat.send_message(user_question, stream=True)
    else:
        full_prompt = context_prompt + f"用户问题：{user_question}"
        response = model.generate_content(full_prompt, stream=True)

    for chunk in response:
        if chunk.text:
            yield chunk.text
