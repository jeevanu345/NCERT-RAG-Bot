"""Optional terminal client for the same NVIDIA NIM retrieval pipeline as the web app."""
import argparse
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'Full stack Project'))
from dotenv import load_dotenv
from fastapi import HTTPException
from api.rag_pipeline import DEFAULT_MODEL, answer_question, resolve_key


def main():
    parser = argparse.ArgumentParser(description='Ask a PDF questions using your NVIDIA NIM key.')
    parser.add_argument('--pdf', required=True, type=Path, help='Path to a text-based textbook PDF')
    args = parser.parse_args()
    load_dotenv(ROOT / 'Full stack Project' / '.env.local')
    try:
        key = resolve_key(None)
        from api.index import process_pdf
        session = str(uuid.uuid5(uuid.NAMESPACE_URL, str(ROOT / 'terminal-workspace')))
        result = process_pdf(args.pdf.read_bytes(), args.pdf.name, session)
        doc = result['document']; messages = []
        print(f"NCERT Tutor · NVIDIA NIM · {doc['name']}\nType exit to finish.")
        while True:
            question = input('\nYou: ').strip()
            if question.lower() in ('exit', 'quit'): break
            if not question: continue
            messages.append({'role':'user', 'content': question[:4000]})
            try:
                result = answer_question(session, messages[-12:], key, os.environ.get('NVIDIA_MODEL', DEFAULT_MODEL), 'explain', [doc['id']])
                print('\nTutor:', result['answer'])
                for source in result['sources']:
                    print(f"[{source['citation']}] {source['name']}, PDF page {source['page']}")
                messages.append({'role':'assistant', 'content': result['answer']})
            except HTTPException as error:
                messages.pop(); print(error.detail)
    except (OSError, HTTPException) as error:
        print(getattr(error, 'detail', str(error)), file=sys.stderr); return 1
    except (EOFError, KeyboardInterrupt):
        print('\nStudy session finished.')
    return 0


if __name__ == '__main__': raise SystemExit(main())
