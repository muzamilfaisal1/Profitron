"""Split research_paper.md into plain-text chunks sized for free plagiarism
and AI-detection checkers.

Outputs two folders:
  chunks_plagiarism/   ~900-word plain text files (fits Smallseotools, Duplichecker, Quetext free)
  chunks_ai/           ~4500-word plain text files (fits GPTZero, ZeroGPT free tiers)
"""
import re
from pathlib import Path

SRC = Path('research_paper.md')
OUT_PLAG = Path('chunks_plagiarism')
OUT_AI = Path('chunks_ai')
OUT_PLAG.mkdir(exist_ok=True)
OUT_AI.mkdir(exist_ok=True)

PLAG_TARGET_WORDS = 900   # under 1000 to fit free tools
AI_TARGET_WORDS = 4500    # under 5000 to fit GPTZero free


def strip_markdown(text):
    # Remove headings, bullets, image refs, hr lines, bold/italic markers
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('---'):
            continue
        if s.startswith('!['):
            continue
        if s.startswith('## ') or s.startswith('### ') or s.startswith('#### '):
            # keep heading text but as plain line
            s = re.sub(r'^#+\s*', '', s)
        if s.startswith('- '):
            s = s[2:]
        # remove inline markdown
        s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
        s = re.sub(r'\*(.+?)\*', r'\1', s)
        s = re.sub(r'`(.+?)`', r'\1', s)
        # strip reference brackets like [n] for cleaner text
        # (leave them in — checkers don't mind)
        lines.append(s)
    # collapse blank-line clusters to one blank line
    out = []
    blank = False
    for l in lines:
        if l == '':
            if not blank:
                out.append('')
            blank = True
        else:
            out.append(l)
            blank = False
    return '\n'.join(out).strip()


def split_by_words(text, target):
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    cur = []
    cur_w = 0
    for p in paragraphs:
        w = len(p.split())
        if cur and cur_w + w > target:
            chunks.append('\n\n'.join(cur))
            cur = []
            cur_w = 0
        cur.append(p)
        cur_w += w
    if cur:
        chunks.append('\n\n'.join(cur))
    return chunks


raw = SRC.read_text()
plain = strip_markdown(raw)

# Plagiarism chunks
plag_chunks = split_by_words(plain, PLAG_TARGET_WORDS)
for i, c in enumerate(plag_chunks, 1):
    fn = OUT_PLAG / f'chunk_{i:02d}.txt'
    fn.write_text(c + '\n')

# AI-detection chunks
ai_chunks = split_by_words(plain, AI_TARGET_WORDS)
for i, c in enumerate(ai_chunks, 1):
    fn = OUT_AI / f'chunk_{i:02d}.txt'
    fn.write_text(c + '\n')

print(f'Plagiarism chunks ({PLAG_TARGET_WORDS}w each): {len(plag_chunks)} files in {OUT_PLAG}/')
for f in sorted(OUT_PLAG.glob('*.txt')):
    w = len(f.read_text().split())
    print(f'  {f.name}  ({w} words)')

print(f'\nAI-detection chunks ({AI_TARGET_WORDS}w each): {len(ai_chunks)} files in {OUT_AI}/')
for f in sorted(OUT_AI.glob('*.txt')):
    w = len(f.read_text().split())
    print(f'  {f.name}  ({w} words)')
