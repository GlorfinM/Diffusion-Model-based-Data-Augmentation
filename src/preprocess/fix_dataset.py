import os
import shutil
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
RAW_DIR = Path("data/raw")
TEMP_DIR = Path("data/temp_fix")
TAR_PATH = TEMP_DIR / "images.tar.gz"
EXTRACT_DIR = TEMP_DIR / "images"

def download_file(url, dest_path):
    """Download file with progress bar"""
    if dest_path.exists():
        print(f"Temporary file exists: {dest_path}, skipping download.")
        return

    print(f"Downloading dataset: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1KB

    with open(dest_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def fix_dataset():
    # 1. Prepare temp directory
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Download official dataset
    try:
        download_file(DATASET_URL, TAR_PATH)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Check network connection.")
        return

    # 3. Extract files
    print("Extracting dataset...")
    if not EXTRACT_DIR.exists():
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall(path=TEMP_DIR)
    
    source_images_dir = TEMP_DIR / "images"
    if not source_images_dir.exists():
        print(f"Extraction structure invalid, not found: {source_images_dir}")
        return

    # 4. Verify and fix
    print(f"Verifying against data/raw ...")
    
    all_source_files = list(source_images_dir.glob("*.jpg"))
    total_files = len(all_source_files)
    print(f"Source contains {total_files} images.")

    fixed_count = 0
    skipped_count = 0

    for src_file in tqdm(all_source_files, desc="Verifying & Fixing"):
        # Parse filename for class
        # Format: Class_Name_Number.jpg
        filename = src_file.name
        
        try:
            class_name = filename.rsplit("_", 1)[0]
        except IndexError:
            continue

        target_dir = RAW_DIR / class_name
        target_file = target_dir / filename

        # Check existence
        if not target_file.exists():
            # Missing file found, fix it
            target_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src_file, target_file)
            fixed_count += 1
        else:
            skipped_count += 1

    # 5. Summary and cleanup
    print("\n" + "="*40)
    print(f"Verification complete.")
    print(f"   - Matches: {skipped_count}")
    print(f"   - Fixed: {fixed_count}")
    print("="*40)

    # Cleanup temp files
    print("Cleaning up temporary files...")
    try:
        if TAR_PATH.exists():
            os.remove(TAR_PATH)
        if source_images_dir.exists():
            shutil.rmtree(source_images_dir)
        if TEMP_DIR.exists():
            TEMP_DIR.rmdir()
        print("Cleanup complete.")
    except Exception as e:
        print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    fix_dataset()