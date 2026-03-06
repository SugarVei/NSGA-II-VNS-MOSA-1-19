"""
文献管理路由
- PDF 文件上传
- 文献列表查询
- PDF 解析触发
"""

import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import UPLOAD_DIR
from app.services.pdf_parser import parse_pdf, PaperData

router = APIRouter(prefix="/api/papers", tags=["文献管理"])

# 内存缓存：存储已解析的论文数据，避免重复解析
_parsed_cache: dict[str, PaperData] = {}


@router.post("/upload")
async def upload_paper(file: UploadFile = File(...)):
    """
    上传 PDF 文件到服务器。
    支持拖拽上传，文件保存到 uploads/ 目录。
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"filename": file.filename, "message": "上传成功"}


@router.get("/list")
async def list_papers():
    """返回已上传的 PDF 文件列表"""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    return {"papers": files}


@router.get("/parse/{filename}")
async def parse_paper(filename: str):
    """
    解析指定 PDF 文件，返回带坐标信息的结构化数据。
    结果会被缓存，相同文件不会重复解析。
    """
    # 检查缓存
    if filename in _parsed_cache:
        return _parsed_cache[filename]

    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")

    paper_data = parse_pdf(file_path)
    _parsed_cache[filename] = paper_data

    return paper_data


@router.delete("/{filename}")
async def delete_paper(filename: str):
    """删除指定 PDF 文件"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    _parsed_cache.pop(filename, None)
    return {"message": "删除成功"}
