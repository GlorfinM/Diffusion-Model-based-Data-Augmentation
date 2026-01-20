import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Diffusers 库
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# === Configuration ===
# Base model: SD 1.5
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
# IP-Adapter Repo
IP_ADAPTER_REPO = "h94/IP-Adapter"
# Weights file
IP_ADAPTER_BIN = "ip-adapter_sd15.bin"

# Negative prompt
NEGATIVE_PROMPT = "deformed, distorted, disfigured, bad anatomy, bad eyes, extra limbs, blurry, low quality, watermark, text, ugly, mutation"

class IPAdapterAugmentor:
    def __init__(self, gpu_id=0, adapter_scale=0.6):
        """
        Initialize IP-Adapter Augmentor
        :param gpu_id: GPU ID
        :param adapter_scale: Image prompt weight (0.0 - 1.0)
        """
        # Initialize device
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Initializing IP-Adapter model on device: {self.device}...")
        
        # Load SD 1.5
        self.pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # Switch to DPM++ Scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # Load IP-Adapter weights
        print("Loading IP-Adapter components...")
        self.pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_BIN)
        
        # Set Scale
        self.pipe.set_ip_adapter_scale(adapter_scale)
        
        # Move to GPU
        self.pipe = self.pipe.to(self.device)
        
        # # Memory optimization
        # try:
        #     self.pipe.enable_xformers_memory_efficient_attention()
        # except Exception:
        #     pass

    def augment_image(self, image_path, output_dir, num_vars=2, target_size=(512, 512)):
        """
        Process single image variation
        """
        try:
            # Read and preprocess image
            org_image = Image.open(image_path).convert("RGB")
            # Resize for prompt input only
            ip_image = org_image.resize(target_size) 

            # Get class name
            class_name = image_path.parent.name.replace("_", " ")
            
            # Build Prompt
            prompt = f"a photo of a {class_name}, high quality, realistic, detailed fur"

            # Check if already generated
            if (output_dir / f"{image_path.stem}_var0.jpg").exists():
                return 0

            # Inference
            with torch.autocast("cuda"):
                images = self.pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    ip_adapter_image=ip_image,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    num_images_per_prompt=num_vars,
                    width=target_size[0],
                    height=target_size[1],
                ).images

            # Save images
            for i, img in enumerate(images):
                save_path = output_dir / f"{image_path.stem}_var{i}.jpg"
                img.save(save_path)
            
            return len(images)

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            return 0

def main():
    parser = argparse.ArgumentParser(description="Image variation augmentation using IP-Adapter")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw image directory")
    parser.add_argument("--output_dir", type=str, default="data/augmented/ip_adapter_var", help="Output directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--num_vars", type=int, default=2, help="Number of variations per image")
    parser.add_argument("--scale", type=float, default=0.6, help="IP-Adapter Scale (0.0-1.0)")
    args = parser.parse_args()

    # Initialize Augmentor
    augmentor = IPAdapterAugmentor(gpu_id=args.gpu_id, adapter_scale=args.scale)

    raw_path = Path(args.raw_dir)
    output_path = Path(args.output_dir)

    # Scan images
    all_images = list(raw_path.rglob("*.jpg"))
    print(f"Found {len(all_images)} raw images")

    total_generated = 0
    pbar = tqdm(all_images, desc=f"IP-Adapter Variation on GPU {args.gpu_id}")
    
    for img_file in pbar:
        # Prepare output directory
        rel_path = img_file.relative_to(raw_path)
        class_output_dir = output_path / rel_path.parent
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # Execute augmentation
        count = augmentor.augment_image(
            img_file, 
            class_output_dir, 
            num_vars=args.num_vars
        )
        total_generated += count
        
        pbar.set_postfix({"New Images": total_generated})

    print(f"Task complete! Total generated: {total_generated}")

if __name__ == "__main__":
    main()