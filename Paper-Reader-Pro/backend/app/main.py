"""
Paper-Reader-Pro 后端入口
==========================
FastAPI 应用，提供 PDF 解析、翻译、AI 分析等 API 服务。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import papers, translate, analyze
import os

app = FastAPI(
    title="Paper-Reader-Pro API",
    description="AI 辅助学术文献阅读器后端服务",
    version="1.0.0",
)

# 跨域配置：允许前端开发服务器访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(papers.router)
app.include_router(translate.router)
app.include_router(analyze.router)


# 挂载 uploads 目录为静态文件，供前端 PDF.js 直接加载 PDF
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/")
async def root():
    return {"message": "Paper-Reader-Pro API 运行中"}
