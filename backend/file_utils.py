import os
from typing import List, Dict
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text
from unstructured.partition.csv import partition_csv
from unstructured.partition.xlsx import partition_xlsx
from PIL import Image
import pytesseract
import io

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def ocr_image_file(file_path: str) -> List[Dict]:
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return [{'text': chunk, 'meta': {}} for chunk in chunk_text(text)]

def ocr_pdf_file(file_path: str) -> List[Dict]:
    from pdf2image import convert_from_path
    chunks = []
    images = convert_from_path(file_path)
    for page_num, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        for chunk in chunk_text(text):
            chunks.append({'text': chunk, 'meta': {'page': page_num+1}})
    return chunks

def parse_file(file_path: str) -> List[Dict]:
    ext = os.path.splitext(file_path)[-1].lower()
    chunks = []
    if ext == '.pdf':
        elements = partition_pdf(filename=file_path)
        if not elements or all(not el.text.strip() for el in elements):
            try:
                return ocr_pdf_file(file_path)
            except Exception:
                raise ValueError("PDF could not be parsed or OCR'd.")
        for el in elements:
            for chunk in chunk_text(el.text):
                chunks.append({'text': chunk, 'meta': {}})
    elif ext == '.docx':
        elements = partition_docx(filename=file_path)
        for el in elements:
            for chunk in chunk_text(el.text):
                chunks.append({'text': chunk, 'meta': {}})
    elif ext == '.txt':
        elements = partition_text(filename=file_path)
        for el in elements:
            for chunk in chunk_text(el.text):
                chunks.append({'text': chunk, 'meta': {}})
    elif ext == '.csv':
        elements = partition_csv(filename=file_path)
        for el in elements:
            for chunk in chunk_text(el.text):
                chunks.append({'text': chunk, 'meta': {}})
    elif ext == '.xlsx':
        elements = partition_xlsx(filename=file_path)
        for el in elements:
            for chunk in chunk_text(el.text):
                chunks.append({'text': chunk, 'meta': {}})
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return ocr_image_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return chunks
