"""
Submission Packaging Script for Amazon ML Challenge 2026.
Creates TEAM_NAME_submission.zip according to the exact required package structure:

TEAM_NAME_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── load_data.py
│       │   ├── preprocessing.py
│       │   ├── normalization.py
│       │   ├── blocking.py
│       │   ├── similarity.py
│       │   ├── features.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── predict.py
│       │   ├── postprocess.py
│       │   ├── submission.py
│       │   └── run_pipeline.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CODE_DIR = PROJECT_ROOT / "business_entity_resolution"
DOC_FILE = PROJECT_ROOT / "Documentation_template.md"
ZIP_NAME = "TEAM_NAME_submission.zip"
ZIP_PATH = PROJECT_ROOT / ZIP_NAME


def create_submission_zip():
    print(f"[Packaging] Preparing submission package at: {ZIP_PATH}")

    # Remove existing zip if present
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        # 1. Add output/ directory
        if OUTPUT_DIR.exists():
            for f in sorted(OUTPUT_DIR.glob("*.tsv")):
                arcname = f"output/{f.name}"
                zipf.write(f, arcname=arcname)
                print(f"  [+] Added: {arcname}")
        else:
            print(f"  [!] Warning: Output directory not found at: {OUTPUT_DIR}")

        # 2. Add code/business_entity_resolution/ directory
        if CODE_DIR.exists():
            for root, dirs, files in os.walk(CODE_DIR):
                # Exclude __pycache__, .git, and models/ checkpoints from code zip to keep archive lightweight
                dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".pytest_cache")]
                for file in sorted(files):
                    if file.endswith((".pyc", ".DS_Store", ".tmp", ".pkl")):
                        continue
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(CODE_DIR)
                    arcname = f"code/business_entity_resolution/{rel_path.as_posix()}"
                    zipf.write(full_path, arcname=arcname)
                    print(f"  [+] Added: {arcname}")
        else:
            print(f"  [!] Warning: Code directory not found at: {CODE_DIR}")

        # 3. Add Documentation_template.md
        if DOC_FILE.exists():
            zipf.write(DOC_FILE, arcname="Documentation_template.md")
            print("  [+] Added: Documentation_template.md")
        else:
            print(f"  [!] Warning: Documentation file not found at: {DOC_FILE}")

    zip_size_kb = ZIP_PATH.stat().st_size / 1024
    print(f"\n[Success] Submission package created: {ZIP_NAME} ({zip_size_kb:.1f} KB)")


if __name__ == "__main__":
    create_submission_zip()
