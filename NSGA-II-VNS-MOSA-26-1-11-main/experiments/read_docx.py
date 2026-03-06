import docx
import sys

def extract_math_model(file_path, out_path):
    doc = docx.Document(file_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            if "Model" in text or "Mathematical" in text or "模型" in text or "约束" in text or "目标函数" in text:
                f.write(text + "\n")
            elif "=" in text or "≤" in text or "≥" in text or "\\sum" in text or "+" in text:
                f.write(text + "\n")

if __name__ == "__main__":
    extract_math_model(r"D:\Users\86181\Desktop\算法代码\NSGA-II-VNS-MOSA-26-1-11-paper-aligned-audit\论文写作有关文件\初稿-引用完毕.docx", r"D:\Users\86181\Desktop\算法代码\NSGA-II-VNS-MOSA-26-1-11-paper-aligned-audit\NSGA-II-VNS-MOSA-26-1-11-main\docx_output_utf8.txt")
