import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text(docx_path):
    text = []
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            
            WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            PARA = WORD_NAMESPACE + 'p'
            TEXT = WORD_NAMESPACE + 't'
            
            for paragraph in tree.iter(PARA):
                texts = [node.text for node in paragraph.iter(TEXT) if node.text]
                if texts:
                    text.append(''.join(texts))
        return '\n'.join(text)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    docx_file = r'D:\Users\86181\Desktop\算法代码\NSGA-II-VNS-MOSA-26-1-11-paper-aligned-audit\论文写作有关文件\初稿-引用完毕.docx'
    out_file = r'D:\Users\86181\Desktop\算法代码\NSGA-II-VNS-MOSA-26-1-11-paper-aligned-audit\extracted_paper.txt'
    
    content = extract_text(docx_file)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully extracted {len(content)} characters to {out_file}.")
