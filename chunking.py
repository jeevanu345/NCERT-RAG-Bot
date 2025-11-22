from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def create_chunks_from_text(text_path):
    """
    Reads a text file and splits it into smaller, overlapping chunks.
    """
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    # The RecursiveCharacterTextSplitter is smart; it tries to split on
    # different characters to keep sentences and paragraphs together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    chunks = text_splitter.split_text(text)
    print(f"Created {len(chunks)} text chunks.")
    return chunks

if __name__ == "__main__":
    text_file = "ncert_text.txt"
    if os.path.exists(text_file):
        chunks = create_chunks_from_text(text_file)
        # You can inspect the first few chunks to see if the process worked
        # for chunk in chunks[:5]:
        #     print(f"--- Chunk Start ---\n{chunk}\n--- Chunk End ---")
    else:
        print(f"Error: The text file '{text_file}' was not found.")