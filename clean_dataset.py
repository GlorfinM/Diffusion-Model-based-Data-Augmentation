import argparse
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Prevent PIL DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None

def check_file_integrity(file_path):
    """
    Dual check:
    1. Check if file size is 0
    2. Check if PIL can verify content
    Returns: (is_valid, error_reason)
    """
    # Check 1: Zero byte file
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "Empty file (0 bytes)"
    except OSError:
        return False, "File inaccessible"

    # Check 2: Image header/data corruption
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify file header and structure
    except Exception as e:
        return False, f"Corrupted ({str(e)})"

    return True, "OK"

def main():
    parser = argparse.ArgumentParser(description="Dataset corruption cleaner")
    parser.add_argument("--target_dir", type=str, default="data/augmented", help="Root directory to scan")
    parser.add_argument("--delete", action="store_true", help="[DANGER] Enable to actually delete files")
    args = parser.parse_args()

    root_path = Path(args.target_dir)
    if not root_path.exists():
        print(f"❌ Directory not found: {root_path}")
        return

    print(f"🔍 Scanning directory: {root_path}")
    if args.delete:
        print("⚠️  [WARNING] DELETE MODE ENABLED! Corrupted files will be removed!")
    else:
        print("🛡️  [INFO] DRY RUN MODE. No files will be deleted.")

    # Recursively find all .jpg / .png files
    extensions = ['*.jpg', '*.jpeg', '*.png']
    all_files = []
    for ext in extensions:
        all_files.extend(list(root_path.rglob(ext)))
    
    print(f"📄 Found {len(all_files)} images, checking integrity...")

    bad_files = []
    
    # Use tqdm for progress
    for file_path in tqdm(all_files, desc="Checking integrity"):
        is_valid, reason = check_file_integrity(file_path)
        
        if not is_valid:
            bad_files.append((file_path, reason))
            # Delete if enabled
            if args.delete:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"\n❌ Delete failed {file_path}: {e}")

    # === Summary Report ===
    print("\n" + "="*50)
    print(f"📊 Scan Report - {root_path}")
    print("="*50)
    print(f"✅ Valid files: {len(all_files) - len(bad_files)}")
    print(f"❌ Corrupted files: {len(bad_files)}")
    
    if len(bad_files) > 0:
        print("\n[Corrupted File Details]")
        # Print only first 10 to avoid clutter
        for i, (fp, reason) in enumerate(bad_files):
            status = "Deleted" if args.delete else "Not Deleted"
            print(f"  {i+1}. [{status}] {reason}: {fp.name}")
            if i >= 9:
                print(f"  ... and {len(bad_files)-10} more files")
                break
        
        print("-" * 50)
        if not args.delete:
            print(f"💡 Found {len(bad_files)} corrupted files. Run with --delete to clean them.")
            print(f"   Example: python clean_dataset.py --target_dir {args.target_dir} --delete")
        else:
            print(f"🗑️  Successfully cleaned {len(bad_files)} corrupted files.")
            print("🚀 You can now re-run the generation script to fill these gaps.")
    else:
        print("✨ Perfect! No corrupted files found.")

if __name__ == "__main__":
    main()