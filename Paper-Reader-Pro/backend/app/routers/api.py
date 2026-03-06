"""
Paper-Reader-Pro 核心 API 路由
================================
整合为两个主接口：
  1. POST /api/upload_and_parse — 上传 PDF → 解析坐标 → 翻译 → 返回双语 JSON
  2. POST /api/chat            — 论文全文上下文 + 用户问题 → Gemini SSE 流式输出

辅助接口：
  - GET  /api/papers/list      — 获取已上传的文献列表
  - POST /api/config           — 保存 API Key
  - GET  /api/config           — 查询配置状态
"""

import os
import json
import shutil
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import UPLOAD_DIR, api_keys
from app.services.pdf_parser import (
    parse_pdf, get_translatable_blocks, get_full_text, PaperData
)
from app.services.translator import translate_blocks
from app.services.analyzer import analyze_stream

router = APIRouter(prefix="/api", tags=["核心接口"])

# ============================================================
# 内存缓存：避免同一 PDF 重复解析
# key = filename, value = PaperData
# ============================================================
_parsed_cache: dict[str, PaperData] = {}


# ============================================================
# 1. /api/upload_and_parse
#    上传 PDF → 解析 Bounding Box → 调用 OpenAI 翻译 → 返回结果
# ============================================================

@router.post("/upload_and_parse")
async def upload_and_parse(file: UploadFile = File(...)):
    """
    一站式接口：上传 PDF 文件，自动完成解析和翻译。

    处理流程：
    ┌────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌──────────────┐
    │ 接收 PDF   │ →  │ PyMuPDF 解析     │ →  │ OpenAI 批量翻译   │ →  │ 返回 JSON    │
    │ 保存到磁盘  │    │ 提取 BBox 坐标   │    │ 跳过公式/图片区域  │    │ 含坐标+译文  │
    └────────────┘    └──────────────────┘    └───────────────────┘    └──────────────┘

    返回的 JSON 结构：
    {
      "filename": "paper.pdf",
      "total_pages": 10,
      "pages": [
        {
          "page_number": 1,
          "width": 612.0,
          "height": 792.0,
          "blocks": [
            {
              "block_id": "p1_b0",
              "type": "text",           // text / image / formula
              "bbox": { "x0": 72.0, "y0": 100.5, "x1": 540.0, "y1": 130.2 },
              "original": "English paragraph text...",
              "translated": "中文段落译文...",
              "is_translatable": true
            },
            {
              "block_id": "p1_b3",
              "type": "formula",        // 公式块：不翻译，原样保留
              "bbox": { "x0": 72.0, "y0": 200.0, "x1": 540.0, "y1": 280.0 },
              "original": "min ∑ c_ij x_ij ...",
              "translated": null,       // 公式不翻译
              "is_translatable": false
            }
          ]
        }
      ]
    }
    """
    # ---- 步骤 1：校验并保存文件 ----
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 格式文件")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # ---- 步骤 2：PyMuPDF 解析 PDF，提取文本块 + Bounding Box 坐标 ----
    try:
        paper_data = parse_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {str(e)}")

    # 存入缓存（后续 /api/chat 需要用到全文）
    _parsed_cache[file.filename] = paper_data

    # ---- 步骤 3：逐页调用 OpenAI 翻译可翻译的文本块 ----
    # 按页分组并发翻译，提高效率
    page_translations: dict[int, dict[str, str]] = {}
    # key = page_number, value = { block_id: translated_text }

    if api_keys.openai_key:
        # 为每页创建翻译任务
        async def translate_page(page_num: int):
            blocks = get_translatable_blocks(paper_data, page_number=page_num)
            if not blocks:
                return page_num, {}
            try:
                results = await translate_blocks(blocks)
                # 构建 block_id → 译文 的映射
                mapping = {r["block_id"]: r["translated"] for r in results}
                return page_num, mapping
            except Exception:
                # 单页翻译失败不影响其他页
                return page_num, {}

        # 并发执行所有页面的翻译（限制并发数防止 API 限流）
        semaphore = asyncio.Semaphore(3)  # 最多 3 页同时翻译

        async def throttled_translate(page_num: int):
            async with semaphore:
                return await translate_page(page_num)

        tasks = [
            throttled_translate(page.page_number)
            for page in paper_data.pages
        ]
        results = await asyncio.gather(*tasks)

        for page_num, mapping in results:
            page_translations[page_num] = mapping

    # ---- 步骤 4：组装最终返回的 JSON ----
    # 将翻译结果合并回 paper_data 的结构中
    response_pages = []
    for page in paper_data.pages:
        trans_map = page_translations.get(page.page_number, {})
        response_blocks = []

        for block in page.blocks:
            response_blocks.append({
                "block_id": block.block_id,
                "type": block.type,
                "bbox": block.bbox.model_dump(),
                "original": block.text,
                # 可翻译块填入译文，不可翻译块（公式/图片）设为 None
                "translated": trans_map.get(block.block_id) if block.is_translatable else None,
                "is_translatable": block.is_translatable,
            })

        response_pages.append({
            "page_number": page.page_number,
            "width": page.width,
            "height": page.height,
            "blocks": response_blocks,
        })

    return {
        "filename": file.filename,
        "total_pages": paper_data.total_pages,
        "pages": response_pages,
    }


# ============================================================
# 2. /api/chat
#    接收用户问题 → 注入论文全文上下文 → Gemini SSE 流式输出
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求体"""
    filename: str                          # 当前阅读的 PDF 文件名
    question: str                          # 用户问题
    selected_text: str | None = None       # 可选：用户在 PDF 中划选的文本片段
    chat_history: list[dict] | None = None # 可选：历史对话（多轮对话支持）
    # chat_history 格式: [{"role": "user", "parts": ["..."]}, {"role": "model", "parts": ["..."]}]


@router.post("/chat")
async def chat_with_paper(req: ChatRequest):
    """
    与论文进行 AI 对话，使用 Server-Sent Events (SSE) 流式返回。

    SSE 数据格式：
      - 正常输出: data: {"text": "一段回答文本..."}\n\n
      - 完成标记: data: {"done": true}\n\n
      - 错误信息: data: {"error": "错误描述"}\n\n

    前端通过 fetch + ReadableStream 解析 SSE 事件，
    每收到一个 {"text": "..."} 就追加到对话气泡中，实现打字机效果。
    """
    # ---- 获取论文全文（优先从缓存读取） ----
    if req.filename not in _parsed_cache:
        file_path = os.path.join(UPLOAD_DIR, req.filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"文件 '{req.filename}' 不存在，请先上传"
            )
        try:
            _parsed_cache[req.filename] = parse_pdf(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF 解析失败: {str(e)}")

    paper = _parsed_cache[req.filename]
    paper_text = get_full_text(paper)  # 提取纯文本喂给 Gemini

    # ---- 组装问题：如果有划选文本，拼接到问题前面 ----
    question = req.question
    if req.selected_text:
        question = (
            f"用户在论文中划选了以下文本片段：\n"
            f"「{req.selected_text}」\n\n"
            f"基于以上划选内容，请回答：{question}"
        )

    # ---- SSE 事件生成器 ----
    async def sse_event_generator():
        """
        逐块从 Gemini 获取回答并封装为 SSE 事件。
        使用 analyze_stream() 的异步生成器，每 yield 一个文本块
        就立即推送给前端，减少用户等待。
        """
        try:
            async for chunk in analyze_stream(
                paper_text=paper_text,
                user_question=question,
                chat_history=req.chat_history,
            ):
                # 每个 chunk 封装为 SSE data 事件
                payload = json.dumps({"text": chunk}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            # 全部输出完毕，发送结束标记
            yield f"data: {json.dumps({'done': True})}\n\n"

        except ValueError as e:
            # API Key 未配置等业务错误
            error_payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_payload}\n\n"
        except Exception as e:
            # 其他未预期错误
            error_payload = json.dumps(
                {"error": f"AI 分析出错: {str(e)}"},
                ensure_ascii=False,
            )
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁止 Nginx 缓冲 SSE
        },
    )


# ============================================================
# 辅助接口
# ============================================================

@router.get("/papers/list")
async def list_papers():
    """返回已上传的 PDF 文件列表"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    return {"papers": files}


@router.delete("/papers/{filename}")
async def delete_paper(filename: str):
    """删除指定 PDF 文件及其缓存"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    _parsed_cache.pop(filename, None)
    return {"message": "删除成功"}


class ConfigRequest(BaseModel):
    """API Key 配置请求"""
    openai_key: str | None = None
    gemini_key: str | None = None


@router.post("/config")
async def save_config(req: ConfigRequest):
    """保存 API Key（运行时生效）"""
    if req.openai_key is not None:
        api_keys.openai_key = req.openai_key
    if req.gemini_key is not None:
        api_keys.gemini_key = req.gemini_key
    return {"message": "配置已保存"}


@router.get("/config")
async def get_config():
    """查询 API Key 配置状态（不返回明文 Key）"""
    return {
        "openai_configured": bool(api_keys.openai_key),
        "gemini_configured": bool(api_keys.gemini_key),
    }
