"""
PDF 解析服务 — 核心模块
=========================
使用 PyMuPDF (fitz) 提取 PDF 中每个文本块的内容及其 Bounding Box 坐标。

## 数据结构设计

解析后返回的核心数据结构为 `PaperData`，包含：

```json
{
  "filename": "paper.pdf",
  "total_pages": 10,
  "pages": [
    {
      "page_number": 1,
      "width": 612.0,        // 页面宽度（PDF 点，1点 = 1/72 英寸）
      "height": 792.0,       // 页面高度
      "blocks": [
        {
          "block_id": "p1_b0",           // 唯一标识：页码_块序号
          "type": "text",                // 块类型：text / image / formula
          "bbox": {
            "x0": 72.0,                  // 左上角 x 坐标
            "y0": 100.5,                 // 左上角 y 坐标
            "x1": 540.0,                 // 右下角 x 坐标
            "y1": 130.2                  // 右下角 y 坐标
          },
          "text": "This is a paragraph.", // 提取的文本内容
          "is_translatable": true        // 是否应参与翻译
        }
      ]
    }
  ]
}
```

## 坐标系说明
- PDF 坐标原点在页面 **左上角**（PyMuPDF 默认）
- x 轴向右，y 轴向下
- 单位为 PDF 点（point），1 point = 1/72 inch
- 前端使用这些坐标进行 `position: absolute` 定位，
  实现中文译文与英文原文的段落级垂直对齐
"""

import fitz  # PyMuPDF
import re
from typing import Optional
from pydantic import BaseModel


# ============================================================
# 数据模型定义
# ============================================================

class BoundingBox(BaseModel):
    """文本块的边界框坐标"""
    x0: float  # 左上角 x
    y0: float  # 左上角 y
    x1: float  # 右下角 x
    y1: float  # 右下角 y


class TextBlock(BaseModel):
    """
    单个文本块，对应 PDF 中的一个段落/标题/公式区域。
    - block_id: 全局唯一标识，格式为 "p{页码}_b{块序号}"
    - type: 块类型，text 表示普通文本，image 表示图片，formula 表示公式/伪代码
    - bbox: 边界框坐标，用于前端精准定位
    - text: 提取的文本内容
    - is_translatable: 标记该块是否应发送给翻译 API
      公式、伪代码、图表标注等设为 False，避免被强行翻译破坏排版
    """
    block_id: str
    type: str  # "text" | "image" | "formula"
    bbox: BoundingBox
    text: str
    is_translatable: bool


class PageData(BaseModel):
    """单页解析结果"""
    page_number: int
    width: float
    height: float
    blocks: list[TextBlock]


class PaperData(BaseModel):
    """整篇论文的解析结果"""
    filename: str
    total_pages: int
    pages: list[PageData]


# ============================================================
# 公式/伪代码检测器
# ============================================================

# 用于识别可能是数学公式或算法伪代码的正则模式
_FORMULA_PATTERNS = [
    # LaTeX 风格的数学标记
    r"\\(?:sum|prod|int|frac|sqrt|alpha|beta|gamma|delta|theta|lambda|mu|sigma|omega)",
    r"\\(?:min|max|arg\s*min|arg\s*max|subject\s+to|s\.t\.)",
    # 常见的数学运算符组合（如 ≤, ≥, ∈, ∀, ∃, →）
    r"[≤≥∈∀∃→∞∑∏∫≠±×÷∇∂]",
    # 大量数学符号混合（括号、上下标密集出现）
    r"(?:[_^{}\[\]]{3,})",
]

# 伪代码关键词（连续出现多个则判定为伪代码块）
_PSEUDOCODE_KEYWORDS = [
    "algorithm", "procedure", "function", "while", "for each",
    "if ", "then", "else", "end if", "end while", "end for",
    "return", "input:", "output:", "repeat", "until",
]


def _is_formula_or_pseudocode(text: str) -> bool:
    """
    判断文本块是否为数学公式或算法伪代码。

    判定逻辑：
    1. 文本中匹配到 2 个以上数学公式模式 → 判定为公式
    2. 文本中匹配到 3 个以上伪代码关键词 → 判定为伪代码
    3. 文本中数字和符号占比超过 60% → 可能是公式表格

    返回:
        True  → 该块不应参与翻译
        False → 该块为普通文本，可以翻译
    """
    text_lower = text.lower().strip()

    if not text_lower or len(text_lower) < 5:
        return False

    # 检测数学公式模式
    formula_hits = sum(
        1 for pattern in _FORMULA_PATTERNS
        if re.search(pattern, text)
    )
    if formula_hits >= 2:
        return True

    # 检测伪代码关键词
    pseudo_hits = sum(
        1 for kw in _PSEUDOCODE_KEYWORDS
        if kw in text_lower
    )
    if pseudo_hits >= 3:
        return True

    # 检测数字/符号占比（排除空格后）
    non_space = text.replace(" ", "")
    if len(non_space) > 20:
        alpha_count = sum(1 for c in non_space if c.isalpha())
        alpha_ratio = alpha_count / len(non_space)
        # 字母占比低于 40%，说明大部分是数字/符号 → 可能是公式
        if alpha_ratio < 0.4:
            return True

    return False


# ============================================================
# 核心解析函数
# ============================================================

def parse_pdf(file_path: str) -> PaperData:
    """
    解析 PDF 文件，提取所有文本块及其 Bounding Box 坐标。

    处理流程：
    1. 打开 PDF 文件
    2. 逐页遍历，使用 page.get_text("dict") 获取结构化数据
    3. 对每个文本块（block）：
       a. 提取 bbox 坐标 (x0, y0, x1, y1)
       b. 合并块内所有 span 的文本
       c. 调用 _is_formula_or_pseudocode() 判定是否为公式/伪代码
       d. 生成唯一 block_id
    4. 返回结构化的 PaperData 对象

    参数:
        file_path: PDF 文件在服务器上的路径

    返回:
        PaperData 对象，包含所有页面及文本块信息
    """
    doc = fitz.open(file_path)
    pages: list[PageData] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_width = page.rect.width
        page_height = page.rect.height

        # 使用 "dict" 模式提取结构化内容
        # 返回的字典包含 blocks 列表，每个 block 有 bbox 和 lines
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        blocks: list[TextBlock] = []
        block_counter = 0

        for block in page_dict.get("blocks", []):
            block_type = block.get("type", 0)
            # type=0 是文本块，type=1 是图片块
            bbox_raw = block.get("bbox", (0, 0, 0, 0))

            bbox = BoundingBox(
                x0=round(bbox_raw[0], 2),
                y0=round(bbox_raw[1], 2),
                x1=round(bbox_raw[2], 2),
                y1=round(bbox_raw[3], 2),
            )

            if block_type == 1:
                # 图片块：标记为不可翻译，保留位置信息
                blocks.append(TextBlock(
                    block_id=f"p{page_idx + 1}_b{block_counter}",
                    type="image",
                    bbox=bbox,
                    text="[图片]",
                    is_translatable=False,
                ))
                block_counter += 1
                continue

            # 文本块：遍历所有行和 span，合并文本
            full_text = ""
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                # 每行文本之间用空格连接（英文段落内换行通常是软换行）
                if full_text and line_text:
                    # 如果上一行末尾是连字符，去掉连字符直接拼接
                    if full_text.endswith("-"):
                        full_text = full_text[:-1] + line_text
                    else:
                        full_text += " " + line_text
                else:
                    full_text = line_text

            full_text = full_text.strip()

            # 跳过空白块
            if not full_text:
                continue

            # 判定文本类型
            if _is_formula_or_pseudocode(full_text):
                block_type_str = "formula"
                translatable = False
            else:
                block_type_str = "text"
                translatable = True

            blocks.append(TextBlock(
                block_id=f"p{page_idx + 1}_b{block_counter}",
                type=block_type_str,
                bbox=bbox,
                text=full_text,
                is_translatable=translatable,
            ))
            block_counter += 1

        pages.append(PageData(
            page_number=page_idx + 1,
            width=round(page_width, 2),
            height=round(page_height, 2),
            blocks=blocks,
        ))

    doc.close()

    return PaperData(
        filename=file_path.split("/")[-1],
        total_pages=len(pages),
        pages=pages,
    )


def get_translatable_blocks(paper: PaperData, page_number: Optional[int] = None) -> list[TextBlock]:
    """
    从解析结果中提取所有可翻译的文本块。

    参数:
        paper: 已解析的论文数据
        page_number: 可选，指定页码。为 None 时返回全部页面。

    返回:
        可翻译的 TextBlock 列表
    """
    result = []
    for page in paper.pages:
        if page_number is not None and page.page_number != page_number:
            continue
        for block in page.blocks:
            if block.is_translatable:
                result.append(block)
    return result


def get_full_text(paper: PaperData) -> str:
    """
    提取论文全文纯文本（用于喂给分析模型作为上下文）。

    返回:
        拼接后的全文字符串，每个块之间用双换行分隔
    """
    texts = []
    for page in paper.pages:
        for block in page.blocks:
            if block.text and block.text != "[图片]":
                texts.append(block.text)
    return "\n\n".join(texts)
