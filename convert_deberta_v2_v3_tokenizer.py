import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--python_path', type=str, required=True)
args = ap.parse_args()


transformers_path = Path(args.python_path) / "site-packages" / "transformers"
input_dir = Path("./deberta_v2_v3_tokenizer")
convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path/filename
    if filepath.exists():
        filepath.unlink()
    shutil.copy(input_dir/filename, filepath)
