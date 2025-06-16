"""
Script to extract text content from PDF files
"""

import sys
from pathlib import Path

import PyPDF2


def extract_pdf_text(pdf_path):
    """Extract text content from a PDF file"""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""

            print(f"PDF has {len(pdf_reader.pages)} pages")
            print("-" * 50)

            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                text_content += f"\n--- PAGE {page_num + 1} ---\n"
                text_content += text
                text_content += "\n"

            return text_content

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


if __name__ == "__main__":
    pdf_path = "/Users/sanjeevadodlapati/Downloads/Repos/ChemML/docs/assets/7DayRoadmap_MLforChemistry.pdf"

    if Path(pdf_path).exists():
        print(f"Reading PDF: {pdf_path}")
        content = extract_pdf_text(pdf_path)

        if content:
            print(content)
        else:
            print("Failed to extract content from PDF")
    else:
        print(f"PDF file not found: {pdf_path}")
