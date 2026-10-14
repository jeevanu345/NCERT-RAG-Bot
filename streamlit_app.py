"""Compatibility launcher: the study interface now runs in the Next.js web app."""
from pathlib import Path
import subprocess

if __name__ == '__main__':
    print('The study interface has moved to the web app. Starting NCERT Tutor…', flush=True)
    raise SystemExit(subprocess.call(['npm', 'run', 'dev'], cwd=Path(__file__).resolve().parent))
