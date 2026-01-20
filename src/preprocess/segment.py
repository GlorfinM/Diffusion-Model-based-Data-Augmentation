import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile

# Prevent image truncation errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate background masks using Rembg (U2Net)")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data/raw", 
        help="Root directory for raw images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/masks", 
        help="Root directory for output masks"
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0, 
        help="GPU ID to use"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="u2net", 
        choices=["u2net", "u2netp", "u2net_human_seg"],
        help="Rembg model to use"
    )
    
    return parser.parse_args()

def process_segmentation(args):
    # Set GPU environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Device: GPU {args.gpu_id}")

    try:
        from rembg import new_session, remove
    except ImportError:
        print("Error: rembg not found. Install with: pip install 'rembg[gpu]'")
        return

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print(f"Error: Input directory not found: {input_root}")
        return

    # Scan for images
    print(f"Scanning images in {input_root}...")
    extensions = ['*.jpg', '*.jpeg', '*.png']
    all_files = []
    for ext in extensions:
        all_files.extend(list(input_root.rglob(ext)))
    
    if not all_files:
        print("No images found.")
        return
        
    print(f"Found {len(all_files)} images.")

    # Initialize model session
    print(f"Loading model '{args.model}' on GPU...")
    start_time = time.time()
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = new_session(model_name=args.model, providers=providers)
    except Exception as e:
        print(f"Model load failed: {e}")
        print("Check onnxruntime-gpu installation.")
        return

    # Process images
    success_count = 0
    skip_count = 0
    error_count = 0

    pbar = tqdm(all_files, desc="Processing", unit="img")
    
    for img_path in pbar:
        try:
            rel_path = img_path.relative_to(input_root)
            out_path = output_root / rel_path.with_suffix('.png')
            
            if out_path.exists():
                skip_count += 1
                continue
            
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(img_path, 'rb') as i:
                input_data = i.read()
                
            output_data = remove(input_data, session=session, only_mask=True)
            
            with open(out_path, 'wb') as o:
                o.write(output_data)
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            pbar.write(f"Error processing {img_path.name}: {str(e)}")

    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*40)
    print(f"Completed in {duration:.2f}s")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {error_count}")
    print(f"Output: {output_root}")
    print("="*40)

if __name__ == "__main__":
    args = parse_args()
    process_segmentation(args)
