"""
分析路由
- 基于论文全文上下文的 AI 对话
- 支持 SSE 流式输出（打字机效果）和划选文本分析
"""

import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.routers.papers import _parsed_cache
from app.services.pdf_parser import parse_pdf, get_full_text
from app.services.analyzer import analyze_stream
from app.config import UPLOAD_DIR, api_keys
import os

router = APIRouter(prefix="/api/analyze", tags=["分析服务"])


class AnalyzeRequest(BaseModel):
    """分析请求体"""
    filename: str
    question: str
    selected_text: str | None = None
    chat_history: list[dict] | None = None


class ConfigRequest(BaseModel):
    """API Key 配置请求"""
    openai_key: str | None = None
    gemini_key: str | None = None


@router.post("/chat")
async def chat_with_paper(req: AnalyzeRequest):
    """
    与论文进行 AI 对话，使用 SSE 流式返回。
    前端通过 fetch + ReadableStream 接收，实现打字机效果。
    """
    if req.filename not in _parsed_cache:
        file_path = os.path.join(UPLOAD_DIR, req.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        _parsed_cache[req.filename] = parse_pdf(file_path)

    paper = _parsed_cache[req.filename]
    paper_text = get_full_text(paper)

    question = req.question
    if req.selected_text:
        question = (
            f"用户划选了以下文本片段：\n"
            f"「{req.selected_text}」\n\n"
            f"基于以上划选内容，{question}"
        )

    async def event_generator():
        """SSE 事件生成器 — 逐块输出 AI 回答"""
        try:
            async for chunk in analyze_stream(
                paper_text=paper_text,
                user_question=question,
                chat_history=req.chat_history,
            ):
                data = json.dumps({"text": chunk}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/config")
async def save_config(req: ConfigRequest):
    """保存 API Key 配置"""
    if req.openai_key is not None:
        api_keys.openai_key = req.openai_key
    if req.gemini_key is not None:
        api_keys.gemini_key = req.gemini_key
    return {"message": "配置已保存"}


@router.get("/config")
async def get_config():
    """获取当前配置状态"""
    return {
        "openai_configured": bool(api_keys.openai_key),
        "gemini_configured": bool(api_keys.gemini_key),
    }
