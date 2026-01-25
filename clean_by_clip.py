import argparse
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import os

# === Configuration ===
# Threshold settings:
# 0.8: Very strict, deletes slight variations (high quality, low quantity)
# 0.7: Moderate (Recommended)
# 0.6: Lenient (only deletes completely broken images)
THRESHOLD = 0.8

class CLIPCleaner:
    def __init__(self, gpu_id=0):
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [GPU {gpu_id}] Initializing CLIP Watchdog (openai/clip-vit-base-patch32)...")
        
        # Load CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def calculate_similarity(self, img_path_a, img_path_b):
        try:
            image_a = Image.open(img_path_a).convert("RGB")
            image_b = Image.open(img_path_b).convert("RGB")

            # Preprocess
            inputs = self.processor(images=[image_a, image_b], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # Normalize features
            features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (features[0] @ features[1].T).item()
            return similarity
        
        except Exception as e:
            print(f"⚠️ Read Error: {e}")
            return 0.0

def main():
    parser = argparse.ArgumentParser(description="Clean generated garbage data using CLIP")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw images directory")
    parser.add_argument("--aug_dir", type=str, default="data/augmented/sdedit_opt", help="Generated images directory")
    parser.add_argument("--delete", action="store_true", help="[DANGER] Enable to actually delete files")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    cleaner = CLIPCleaner(gpu_id=args.gpu_id)
    
    raw_root = Path(args.raw_dir)
    aug_root = Path(args.aug_dir)
    
    # Find all generated images
    aug_images = list(aug_root.rglob("*.jpg"))
    print(f"🔍 Scanned {len(aug_images)} augmented images, starting CLIP audit (Threshold: {THRESHOLD})...")

    deleted_count = 0
    bad_files = []

    for aug_file in tqdm(aug_images, desc="Auditing"):
        # 1. Find corresponding original image
        # Assuming generated file: Beagle_01_sketch.jpg
        # Original file should be: Beagle_01.jpg
        # We need to remove the suffix (_sketch, _oil)
        
        # Simple method: try removing the last underscore suffix
        stem = aug_file.stem # Beagle_01_sketch
        original_stem = "_".join(stem.split("_")[:-1]) # Beagle_01
        
        # Look for original image in raw directory (maintaining structure)
        rel_path = aug_file.parent.relative_to(aug_root) # specific_class/
        raw_file = raw_root / rel_path / f"{original_stem}.jpg"

        if not raw_file.exists():
            # Try alternative naming logic if needed
            # e.g., if original filenames contain underscores
            continue

        # 2. Calculate similarity
        score = cleaner.calculate_similarity(raw_file, aug_file)
        
        # 3. Judge
        if score < THRESHOLD:
            bad_files.append((aug_file, score))
            
            if args.delete:
                try:
                    os.remove(aug_file)
                    deleted_count += 1
                except:
                    pass

    print("\n" + "="*50)
    print("📊 Cleaning Report")
    print("="*50)
    if bad_files:
        print(f"❌ Found {len(bad_files)} substandard images (Similarity < {THRESHOLD}):")
        for i, (fp, score) in enumerate(bad_files[:10]):
            status = "Deleted" if args.delete else "Suggested Delete"
            print(f"  {i+1}. [{status}] Score {score:.3f}: {fp.name}")
        if len(bad_files) > 10:
            print(f"  ... and {len(bad_files)-10} others")
            
        if not args.delete:
            print(f"\n💡 Run: python clean_by_clip.py --delete --aug_dir {args.aug_dir} to execute deletion.")
        else:
            print(f"🗑️ Successfully deleted {deleted_count} garbage images.")
    else:
        print("✨ All images meet quality standards!")

if __name__ == "__main__":
    main()