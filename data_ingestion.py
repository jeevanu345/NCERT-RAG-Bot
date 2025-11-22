from pypdf import PdfReader
import os

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

if __name__ == "__main__":
    #pdf_file = "/Users/jeevans/rag/ncert_science_class10.pdf"  # Make sure this file exists in the same directory
    pdf_file = "/Users/jeevans/rag/jesc106 2.pdf"
    output_text_file = "ncert_text.txt"

    if not os.path.exists(pdf_file):
        print(f"Error: The PDF file '{pdf_file}' was not found.")
    else:
        print(f"Starting text extraction from {pdf_file}...")
        extracted_text = extract_text_from_pdf(pdf_file)

        if extracted_text:
            save_text_to_file(extracted_text, output_text_file)
        else:
            print("No text could be extracted. Please check the PDF file.")