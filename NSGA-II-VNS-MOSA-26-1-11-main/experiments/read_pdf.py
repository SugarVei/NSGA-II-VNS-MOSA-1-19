import fitz # PyMuPDF
import sys

def extract_pdf_text(file_path, out_path):
    doc = fitz.open(file_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            f.write(f"--- Page {page_num + 1} ---\n")
            f.write(text + "\n")

if __name__ == "__main__":
    extract_pdf_text(r"D:\Users\86181\Desktop\算法代码\NSGA-II-VNS-MOSA-26-1-11-paper-aligned-audit\论文写作有关文件\MILP_Validation_Scheme.pdf", r"D:\Users\86181\Desktop\算法代码\NSGA-II-VNS-MOSA-26-1-11-paper-aligned-audit\NSGA-II-VNS-MOSA-26-1-11-main\pdf_output_utf8.txt")
