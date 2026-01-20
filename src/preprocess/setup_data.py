import os
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

# === Configuration ===
# Oxford-IIIT Pet Dataset URL
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

# Path definitions
# Structure: data/raw/<class_name>/<image_name>.jpg
ROOT_DIR = Path("data")
RAW_DIR = ROOT_DIR / "raw"

def download_file(url, destination):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    print(f"Downloading dataset: {url}")
    with open(destination, "wb") as file, tqdm(
        desc=destination.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def organize_dataset(source_dir):
    """
    Organize images from source_dir into class-specific subdirectories in RAW_DIR.
    Filename format: Abyssinian_1.jpg -> Class: Abyssinian
    """
    print("Organizing files by class...")
    
    # Get all jpg images
    images = list(source_dir.glob("*.jpg"))
    if not images:
        print(f"No images found in {source_dir}")
        return

    for img_path in tqdm(images, desc="Organizing"):
        # Parse class name from filename
        filename = img_path.name
        if "_" in filename:
            class_name = "_".join(filename.split("_")[:-1])
        else:
            class_name = "Uncategorized"
        
        # Target directory: data/raw/<class_name>
        target_dir = RAW_DIR / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(str(img_path), str(target_dir / filename))

    # Remove source directory
    try:
        shutil.rmtree(source_dir)
        print(f"Cleaned up temporary directory: {source_dir}")
    except OSError as e:
        print(f"Failed to remove temporary directory {source_dir}: {e}")

def setup_oxford_pet_dataset():
    """Main setup function"""
    
    # Check if data is already prepared
    existing_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()] if RAW_DIR.exists() else []
    if len(existing_dirs) > 10:
        print(f"Dataset appears to be ready ({len(existing_dirs)} class folders detected).")
        return

    # Ensure data/raw exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = RAW_DIR / "images.tar.gz"

    # Download
    if not tar_path.exists():
        try:
            download_file(DATA_URL, tar_path)
        except KeyboardInterrupt:
            print("\nDownload interrupted. Cleaning up...")
            if tar_path.exists(): os.remove(tar_path)
            return
        except Exception as e:
            print(f"\nDownload error: {e}")
            return
    else:
        print(f"Archive found, skipping download.")

    # Extract
    extract_temp_dir = RAW_DIR / "images" 
    
    if extract_temp_dir.exists():
        shutil.rmtree(extract_temp_dir)

    print("Extracting...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=RAW_DIR)
    except tarfile.TarError as e:
        print(f"Extraction failed: {e}")
        return

    # Organize
    if extract_temp_dir.exists():
        organize_dataset(extract_temp_dir)
    else:
        print("Error: 'images' folder not found after extraction.")
        return

    # Clean up archive
    print("Removing archive...")
    os.remove(tar_path)
    
    # Final stats
    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    total_images = sum([len(list(d.glob("*.jpg"))) for d in class_dirs])
    print(f"Setup complete.")
    print(f"   - Classes: {len(class_dirs)}")
    print(f"   - Total images: {total_images}")
    print(f"   - Path: {RAW_DIR}")

if __name__ == "__main__":
    setup_oxford_pet_dataset()