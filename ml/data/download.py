import os
import subprocess
from pathlib import Path

DATASETS = {
    "aptos2019": {
        "source": "kaggle",
        "competition": "aptos2019-blindness-detection",
        "images": 5590,
        "classes": 5,
        "use": "primary DR training",
    },
    "eyepacs": {
        "source": "kaggle",
        "competition": "diabetic-retinopathy-detection",
        "images": 88702,
        "classes": 5,
        "use": "large-scale DR training",
    },
    "messidor2": {
        "source": "https://www.adcis.net/en/third-party/messidor2/",
        "images": 1748,
        "classes": 5,
        "use": "external validation",
    },
    "idrid": {
        "source": "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid",
        "images": 516,
        "classes": 5,
        "use": "Indian data, pixel-level annotations, validation",
    },
    "refuge": {
        "source": "https://refuge.grand-challenge.org/",
        "images": 1200,
        "classes": "segmentation masks",
        "use": "optic disc/cup segmentation (glaucoma CDR)",
    },
    "odir5k": {
        "source": "https://odir2019.grand-challenge.org/",
        "images": 5000,
        "classes": "multi-label (8 conditions)",
        "use": "multi-disease training (AMD, glaucoma, etc.)",
    },
}

def download_kaggle(competition: str, dest: Path):
    """Download from Kaggle CLI. Requires ~/.kaggle/kaggle.json"""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", competition, "-p", str(dest)
    ], check=True)
    for zf in dest.glob("*.zip"):
        subprocess.run(["unzip", "-o", str(zf), "-d", str(dest)], check=True)
        zf.unlink()
