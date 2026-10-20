"""
Model Downloader Utility for Real-Time Age and Gender Detection System.

This script verifies the existence of all 6 required deep learning model files
in the `models/` directory and downloads any missing files automatically from
verified open-source repositories.
"""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODEL_URLS = {
    "opencv_face_detector.pbtxt": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/opencv_face_detector.pbtxt"
    ),
    "opencv_face_detector_uint8.pb": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/opencv_face_detector_uint8.pb"
    ),
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/age_net.caffemodel"
    ),
    "gender_deploy.prototxt": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_deploy.prototxt"
    ),
    "gender_net.caffemodel": (
        "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_net.caffemodel"
    ),
}


def download_progress_hook(block_num, block_size, total_size):
    """Prints a clean CLI progress bar during file downloads."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Progress: {percent:6.2f}% ({mb_downloaded:5.2f} MB / {mb_total:5.2f} MB)")
    else:
        kb_downloaded = downloaded / 1024
        sys.stdout.write(f"\r  Downloaded: {kb_downloaded:7.2f} KB")
    sys.stdout.flush()


def ensure_models_exist(models_dir: str = MODELS_DIR) -> bool:
    """
    Checks for the presence of all required model files and downloads missing ones.

    Returns:
        bool: True if all models are present and valid, False otherwise.
    """
    os.makedirs(models_dir, exist_ok=True)
    all_ok = True

    print("==================================================")
    print(" Verifying Age & Gender Detection Model Files")
    print(f" Target Directory: {models_dir}")
    print("==================================================")

    for filename, url in MODEL_URLS.items():
        file_path = os.path.join(models_dir, filename)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"[FOUND] {filename} ({size_mb:.2f} MB)")
        else:
            print(f"[MISSING] {filename} -> Downloading from source...")
            try:
                opener = urllib.request.build_opener()
                opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")]
                urllib.request.install_opener(opener)
                urllib.request.urlretrieve(url, file_path, reporthook=download_progress_hook)
                sys.stdout.write("\n")
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"[SUCCESS] Downloaded {filename} ({size_mb:.2f} MB)")
            except Exception as exc:
                sys.stdout.write("\n")
                print(f"[ERROR] Failed to download {filename}: {exc}")
                all_ok = False

    print("==================================================")
    if all_ok:
        print("[SUCCESS] All required model files are ready!")
    else:
        print("[WARNING] One or more model files could not be downloaded.")
        print("Please check your internet connection or manually place missing files.")
    print("==================================================")
    return all_ok


if __name__ == "__main__":
    success = ensure_models_exist()
    sys.exit(0 if success else 1)
