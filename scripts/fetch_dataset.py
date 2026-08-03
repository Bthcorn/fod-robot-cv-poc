"""Download the FOD-A Pascal VOC dataset (public Google Drive mirror, ~432MB).

The FOD-A GitHub repo (FOD-UNOmaha/FOD-data) only ships tools/docs, not the
images -- the actual data lives on Google Drive, linked from its README.
"""

import subprocess
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_PATH = DATA_DIR / "fod-a-voc.zip"
VOC_DRIVE_ID = "1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x"  # FOD-A v2.1 Pascal VOC, 300x300
EXTRACT_DIR = DATA_DIR / "fod-a-voc"
VOC_ROOT = EXTRACT_DIR / "FODPascalVOCFormat-V.2.1" / "VOC2007"


def main() -> Path:
    DATA_DIR.mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        subprocess.run(["gdown", VOC_DRIVE_ID, "-O", str(ZIP_PATH)], check=True)
    else:
        print(f"already downloaded: {ZIP_PATH}")

    if not VOC_ROOT.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(EXTRACT_DIR)
    else:
        print(f"already extracted: {VOC_ROOT}")

    n_images = len(list((VOC_ROOT / "JPEGImages").glob("*.jpg")))
    print(f"VOC root: {VOC_ROOT} ({n_images} images)")
    return VOC_ROOT


if __name__ == "__main__":
    main()
