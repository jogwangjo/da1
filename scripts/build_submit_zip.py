#!/usr/bin/env python3
"""submit.zip 패키징 스크립트.

submit/script_baseline.py + model/ + vendor_sonics/ → submit.zip

사용법:
  python scripts/build_submit_zip.py [--script submit/script_baseline.py]
"""

import argparse
import shutil
import zipfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=Path, default=Path("submit/script_baseline.py"),
                    help="제출할 script.py 경로")
    ap.add_argument("--output", type=Path, default=Path("submit.zip"),
                    help="출력 zip 경로")
    ap.add_argument("--submit-dir", type=Path, default=Path("submit"),
                    help="submit 디렉토리")
    args = ap.parse_args()

    submit_dir = args.submit_dir
    model_dir = submit_dir / "model"
    vendor_dir = submit_dir / "vendor_sonics"

    # Validate
    if not args.script.exists():
        raise FileNotFoundError(f"Script not found: {args.script}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    # Remove old zip
    if args.output.exists():
        args.output.unlink()

    # Build zip
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as zf:
        # script.py (핵심!)
        zf.write(args.script, "script.py")
        print(f"  Added: script.py (from {args.script})")

        # model/ 디렉토리
        for p in sorted(model_dir.rglob("*")):
            if p.is_file():
                arcname = f"model/{p.relative_to(model_dir)}"
                zf.write(p, arcname)
        print(f"  Added: model/ ({sum(1 for _ in model_dir.rglob('*') if _.is_file())} files)")

        # vendor_sonics/ 디렉토리 (있으면)
        if vendor_dir.exists():
            for p in sorted(vendor_dir.rglob("*")):
                if p.is_file():
                    arcname = f"vendor_sonics/{p.relative_to(vendor_dir)}"
                    zf.write(p, arcname)
            print(f"  Added: vendor_sonics/ ({sum(1 for _ in vendor_dir.rglob('*') if _.is_file())} files)")

        # requirements.txt (있으면)
        req = submit_dir / "requirements.txt"
        if req.exists():
            zf.write(req, "requirements.txt")
            print(f"  Added: requirements.txt")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"\nCreated: {args.output} ({size_mb:.1f} MB)")

    # Verify
    with zipfile.ZipFile(args.output) as zf:
        names = zf.namelist()
        print(f"Files in zip: {len(names)}")
        assert "script.py" in names, "script.py missing!"
        assert any("pytorch_model.bin" in n for n in names), "model weights missing!"
        print("Verification passed ✓")


if __name__ == "__main__":
    main()
