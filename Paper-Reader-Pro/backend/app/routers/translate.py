"""
翻译路由
- 按页面翻译 PDF 内容
- 返回带坐标的结构化翻译结果
"""

from fastapi import APIRouter, HTTPException
from app.routers.papers import _parsed_cache
from app.services.pdf_parser import parse_pdf, get_translatable_blocks
from app.services.translator import translate_blocks
from app.config import UPLOAD_DIR
import os

router = APIRouter(prefix="/api/translate", tags=["翻译服务"])


@router.get("/{filename}/page/{page_number}")
async def translate_page(filename: str, page_number: int):
    """
    翻译指定页面的所有可翻译文本块。

    返回结构化数据，每个翻译块包含：
    - block_id: 块唯一标识
    - bbox: 原始坐标（用于前端精准定位）
    - original: 英文原文
    - translated: 中文译文
    """
    # 确保文件已解析
    if filename not in _parsed_cache:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        _parsed_cache[filename] = parse_pdf(file_path)

    paper = _parsed_cache[filename]

    if page_number < 1 or page_number > paper.total_pages:
        raise HTTPException(status_code=400, detail="页码超出范围")

    # 提取该页可翻译的文本块
    blocks = get_translatable_blocks(paper, page_number)

    if not blocks:
        return {"page_number": page_number, "translations": []}

    try:
        translations = await translate_blocks(blocks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "page_number": page_number,
        "translations": translations,
    }
