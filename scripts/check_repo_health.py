#!/usr/bin/env python3
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
errors = []

required = [
    "LICENSE", "NOTICE.md", "MODEL_LICENSE.md", "MODEL_LICENSE.vi.md",
    "THIRD_PARTY_NOTICES.md", "CHANGELOG.md",
    "licenses/Mage-Flow-MIT.txt", "licenses/Qwen3-VL-Apache-2.0.txt", "licenses/README.md",
    ".github/CONTRIBUTING.md", ".github/CODE_OF_CONDUCT.md", ".github/SECURITY.md",
    ".github/SUPPORT.md", ".github/CODEOWNERS", ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml", ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml", ".github/ISSUE_TEMPLATE/config.yml",
]

for rel in required:
    if not (ROOT / rel).is_file():
        errors.append(f"missing required repository-health file: {rel}")

model_en = (ROOT / "MODEL_LICENSE.md").read_text(encoding="utf-8") if (ROOT / "MODEL_LICENSE.md").is_file() else ""
model_vi = (ROOT / "MODEL_LICENSE.vi.md").read_text(encoding="utf-8") if (ROOT / "MODEL_LICENSE.vi.md").is_file() else ""

for marker in ["Mage-Flow", "MIT License", "Qwen3-VL", "Apache License 2.0", "does not replace, supersede, or relicense"]:
    if marker not in model_en:
        errors.append(f"MODEL_LICENSE.md missing licensing boundary marker: {marker}")

if "[Tiếng Việt](MODEL_LICENSE.vi.md)" not in model_en:
    errors.append("MODEL_LICENSE.md missing Vietnamese language switch")
if "[English](MODEL_LICENSE.md)" not in model_vi:
    errors.append("MODEL_LICENSE.vi.md missing English language switch")

license_hashes = {}
for rel in ["licenses/Mage-Flow-MIT.txt", "licenses/Qwen3-VL-Apache-2.0.txt"]:
    p = ROOT / rel
    if p.is_file():
        license_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()

if errors:
    for error in errors:
        print(f"[FAIL] {error}")
    sys.exit(1)

print("[PASS] repository community-health files")
print("[PASS] model licensing boundary")
print("[PASS] upstream license copies")
for rel, digest in license_hashes.items():
    print(f"SHA256 {rel} {digest}")
print("REPO_HEALTH_CHECK=PASS")
