#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
errors = []

if "[Tiếng Việt](README.vi.md)" not in (ROOT / "README.md").read_text(encoding="utf-8"):
    errors.append("README.md is missing the Vietnamese language switch")
if "[English](README.md)" not in (ROOT / "README.vi.md").read_text(encoding="utf-8"):
    errors.append("README.vi.md is missing the English language switch")

for en in sorted(DOCS.glob("*.md")):
    if en.name.endswith(".vi.md"):
        continue
    vi = en.with_name(en.stem + ".vi.md")
    if not vi.is_file():
        errors.append(f"missing Vietnamese pair for docs/{en.name}")
    elif f"[Tiếng Việt]({vi.name})" not in en.read_text(encoding="utf-8"):
        errors.append(f"missing Vietnamese switch in docs/{en.name}")

for vi in sorted(DOCS.glob("*.vi.md")):
    en = vi.with_name(vi.name.replace(".vi.md", ".md"))
    if not en.is_file():
        errors.append(f"missing English pair for docs/{vi.name}")
    elif f"[English]({en.name})" not in vi.read_text(encoding="utf-8"):
        errors.append(f"missing English switch in docs/{vi.name}")

link_re = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
for md in [ROOT / "README.md", ROOT / "README.vi.md", *sorted(DOCS.glob("*.md"))]:
    for target in link_re.findall(md.read_text(encoding="utf-8")):
        target = target.strip().split()[0].strip("<>")
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean = target.split("#", 1)[0].split("?", 1)[0]
        if not clean:
            continue
        resolved = (md.parent / clean).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"{md.relative_to(ROOT)}: link escapes repository: {target}")
            continue
        if not resolved.exists():
            errors.append(f"{md.relative_to(ROOT)}: broken relative link: {target}")

if errors:
    for error in errors:
        print(f"[FAIL] {error}")
    sys.exit(1)

print("[PASS] bilingual documentation pairs")
print("[PASS] README language switches")
print("[PASS] relative Markdown links")
print("DOCS_CHECK=PASS")
