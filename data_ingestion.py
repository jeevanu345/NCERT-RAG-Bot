from pypdf import PdfReader
from pathlib import Path
import argparse


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a given PDF file.
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return text

def save_text_to_file(text, output_file):
    """
    Saves the extracted text to a .txt file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully extracted and saved text to {output_file}")


def _resolve_pdf_path(pdf_arg: str | None) -> Path | None:
    if pdf_arg:
        candidate = Path(pdf_arg).expanduser()
        return candidate if candidate.exists() else None

    pdfs_in_repo = sorted(Path(".").glob("*.pdf"))
    if not pdfs_in_repo:
        return None
    return pdfs_in_repo[0]


def main():
    parser = argparse.ArgumentParser(description="Extract text from an NCERT PDF.")
    parser.add_argument("--pdf", help="Path to PDF file. If omitted, first *.pdf in this folder is used.")
    parser.add_argument("--output", default="ncert_text.txt", help="Output text file path.")
    args = parser.parse_args()

    pdf_file = _resolve_pdf_path(args.pdf)
    output_text_file = args.output

    if pdf_file is None:
        print("Error: No PDF found. Add a PDF to this folder or pass --pdf /path/to/file.pdf")
        return

    print(f"Starting text extraction from {pdf_file}...")
    extracted_text = extract_text_from_pdf(str(pdf_file))

    if extracted_text:
        save_text_to_file(extracted_text, output_text_file)
    else:
        print("No text could be extracted. Please check the PDF file.")


if __name__ == "__main__":
    main()
