"""
翻译服务模块
=============
调用 OpenAI API (gpt-4o-mini) 进行学术论文的段落级翻译。
返回结构化 JSON，保留原始 block_id 和坐标信息，供前端精准定位。
"""

import json
from openai import AsyncOpenAI
from app.config import api_keys, TRANSLATION_SYSTEM_PROMPT
from app.services.pdf_parser import TextBlock


async def translate_blocks(blocks: list[TextBlock]) -> list[dict]:
    """
    批量翻译文本块。

    处理逻辑：
    1. 将所有可翻译的文本块组装为带编号的批量翻译请求
    2. 调用 OpenAI API 一次性翻译，减少 API 调用次数
    3. 解析返回的 JSON，将译文与原始 block_id、bbox 对应起来

    参数:
        blocks: 可翻译的 TextBlock 列表

    返回:
        翻译结果列表，每项包含:
        {
            "block_id": "p1_b2",
            "bbox": {"x0": 72.0, "y0": 100.5, "x1": 540.0, "y1": 130.2},
            "original": "English text...",
            "translated": "中文译文..."
        }
    """
    if not blocks:
        return []

    if not api_keys.openai_key:
        raise ValueError("请先在配置中心设置 OpenAI API Key")

    client = AsyncOpenAI(api_key=api_keys.openai_key)

    # 组装批量翻译请求：每个段落带编号，便于解析返回结果
    numbered_texts = []
    for i, block in enumerate(blocks):
        numbered_texts.append(f"[{i}] {block.text}")

    user_prompt = (
        "请翻译以下编号段落。以 JSON 数组格式返回，每项为 "
        '{\"index\": 编号, \"translation\": \"中文译文\"}。\n'
        "只返回 JSON 数组，不要返回其他内容。\n\n"
        + "\n\n".join(numbered_texts)
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,  # 低温度保证翻译一致性
        response_format={"type": "json_object"},
    )

    # 解析 API 返回的 JSON
    raw_content = response.choices[0].message.content
    parsed = json.loads(raw_content)

    # API 可能返回 {"translations": [...]} 或直接返回 [...]
    if isinstance(parsed, list):
        translations = parsed
    elif isinstance(parsed, dict):
        translations = parsed.get("translations", parsed.get("results", []))
    else:
        translations = []

    # 建立索引到译文的映射
    trans_map = {}
    for item in translations:
        idx = item.get("index", -1)
        text = item.get("translation", "")
        trans_map[idx] = text

    # 组装最终结果，保留原始坐标信息
    results = []
    for i, block in enumerate(blocks):
        results.append({
            "block_id": block.block_id,
            "bbox": block.bbox.model_dump(),
            "original": block.text,
            "translated": trans_map.get(i, "[翻译失败]"),
        })

    return results
