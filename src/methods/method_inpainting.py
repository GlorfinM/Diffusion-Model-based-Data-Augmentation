import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps

# Diffusers 库
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler

# === Configuration: Scene Prompts ===
PROMPT_SCENES = {
    "snow": "a photo of a pet on a snowy mountain, winter, cold weather, snow covered ground, high resolution, 8k, realistic texture",
    "beach": "a photo of a pet running on a sandy beach, ocean waves in background, sunny day, blue sky, summer vibes, high quality, 8k",
    "jungle": "a photo of a pet in a tropical jungle, green leaves, rainforest, nature, sunlight filtering through trees, detailed background, 8k",
    "city": "a photo of a pet on a city street, urban environment, blurred city lights, bokeh, modern architecture, street photography, realistic",
    "sunset": "a photo of a pet in a field during golden hour, sunset, warm lighting, lens flare, artistic composition, dreamy atmosphere, 8k"
}

NEGATIVE_PROMPT = "ugly, blurry, low quality, deformed, distorted, bad anatomy, bad proportions, watermark, text, signature, mutation, extra limbs"

class InpaintingAugmentor:
    def __init__(self, gpu_id=0, model_id="runwayml/stable-diffusion-inpainting"):
        # Initialize device
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Inpainting model ({model_id}) on device: {self.device}...")
        
        # Load Pipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # Switch scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # Move to GPU
        self.pipe = self.pipe.to(self.device)
        
        # # Memory optimization
        # try:
        #     self.pipe.enable_xformers_memory_efficient_attention()
        # except Exception:
        #     pass

    def augment_image(self, image_path, mask_path, output_dir, target_size=(512, 512)):
        try:
            init_image = Image.open(image_path).convert("RGB").resize(target_size)
            mask_image = Image.open(mask_path).convert("L").resize(target_size)
            
            # Invert Mask
            mask_image = ImageOps.invert(mask_image)

            generated_count = 0
            
            for scene_name, prompt in PROMPT_SCENES.items():
                save_name = f"{image_path.stem}_{scene_name}.jpg"
                save_path = output_dir / save_name

                if save_path.exists():
                    continue

                # Inference
                with torch.autocast("cuda"):
                    result = self.pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        image=init_image,
                        mask_image=mask_image,
                        width=target_size[0],
                        height=target_size[1],
                        num_inference_steps=25,
                        guidance_scale=7.5
                    ).images[0]

                result.save(save_path)
                generated_count += 1
            
            return generated_count

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            return 0

def main():
    parser = argparse.ArgumentParser(description="Background augmentation using Stable Diffusion Inpainting")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Raw image directory")
    parser.add_argument("--mask_dir", type=str, default="data/masks", help="Mask directory")
    parser.add_argument("--output_dir", type=str, default="data/augmented/inpainting_bg", help="Output directory")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    # Pass GPU ID to class initialization
    augmentor = InpaintingAugmentor(gpu_id=args.gpu_id)

    raw_path = Path(args.raw_dir)
    mask_path = Path(args.mask_dir)
    output_path = Path(args.output_dir)

    # Scan images
    all_images = list(raw_path.rglob("*.jpg"))
    print(f"Found {len(all_images)} raw images")

    total_generated = 0
    pbar = tqdm(all_images, desc=f"Inpainting on GPU {args.gpu_id}")
    
    for img_file in pbar:
        rel_path = img_file.relative_to(raw_path)
        mask_file = mask_path / rel_path.with_suffix(".png")

        if not mask_file.exists():
            continue

        class_output_dir = output_path / rel_path.parent
        class_output_dir.mkdir(parents=True, exist_ok=True)

        count = augmentor.augment_image(img_file, mask_file, class_output_dir)
        total_generated += count
        
        pbar.set_postfix({"New Images": total_generated})

    print(f"Task complete! Total generated: {total_generated}")

if __name__ == "__main__":
    main()