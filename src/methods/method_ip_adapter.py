import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Diffusers 库
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# === 配置区域 ===
# 基础模型: SD 1.5 (IP-Adapter 官方推荐底座)
BASE_MODEL = "runwayml/stable-diffusion-v1-5"
# IP-Adapter 仓库
IP_ADAPTER_REPO = "h94/IP-Adapter"
# 权重文件名
IP_ADAPTER_BIN = "ip-adapter_sd15.bin"

# 负面提示词 (防止畸形)
NEGATIVE_PROMPT = "deformed, distorted, disfigured, bad anatomy, bad eyes, extra limbs, blurry, low quality, watermark, text, ugly, mutation"

class IPAdapterAugmentor:
    def __init__(self, gpu_id=0, adapter_scale=0.6):
        """
        初始化 IP-Adapter 增强器
        :param gpu_id: 指定 GPU ID
        :param adapter_scale: 图像提示词的权重 (0.0 - 1.0)。
                              0.6 是平衡点：既像原图，又有足够的变化。
        """
        # 1. 显式构造设备字符串
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"🚀 正在初始化 IP-Adapter 模型到设备: {self.device}...")
        
        # 2. 加载基础 SD 1.5 模型
        self.pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            safety_checker=None
        )
        
        # 3. 切换到 DPM++ 调度器 (速度快，生成质量高)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # 4. 加载 IP-Adapter 权重
        # 这会自动下载 h94/IP-Adapter 下的 models/ip-adapter_sd15.bin
        print("📥 正在加载 IP-Adapter 组件...")
        self.pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="models", weight_name=IP_ADAPTER_BIN)
        
        # 5. 设置 Scale (关键!)
        # 设置为 0.6，意味着 60% 听图的，40% 听 Text Prompt 的 + 随机噪声
        self.pipe.set_ip_adapter_scale(adapter_scale)
        
        # 6. 移动到 GPU
        self.pipe = self.pipe.to(self.device)
        
        # 显存优化
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    def augment_image(self, image_path, output_dir, num_vars=2, target_size=(512, 512)):
        """
        处理单张图片的变分生成
        """
        try:
            # 读取并预处理图片
            # IP-Adapter 最好接收 512x512 的正方形图片作为 Prompt
            org_image = Image.open(image_path).convert("RGB")
            # 这里的 resize 仅用于作为提示词输入，不影响生成图片的尺寸设置
            ip_image = org_image.resize(target_size) 

            # 获取类别名称 (假设路径结构 data/raw/Abyssinian/img.jpg)
            # 将下划线替换为空格: "German_Shepherd" -> "German Shepherd"
            class_name = image_path.parent.name.replace("_", " ")
            
            # 构建 Prompt: 强迫模型生成对应品种
            prompt = f"a photo of a {class_name}, high quality, realistic, detailed fur"

            # 构建保存路径前缀
            # e.g., data/augmented/ip_adapter_var/Abyssinian/img1_var
            save_prefix = output_dir / image_path.stem

            # 检查是否已经生成过 (简单检查第一张)
            if (output_dir / f"{image_path.stem}_var0.jpg").exists():
                return 0

            # 推理
            with torch.autocast("cuda"):
                # num_images_per_prompt=num_vars 一次生成多张
                images = self.pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    ip_adapter_image=ip_image, # 将原图作为视觉提示
                    num_inference_steps=30,    # IP-Adapter 需要稍多步数保证细节
                    guidance_scale=7.5,
                    num_images_per_prompt=num_vars,
                    width=target_size[0],
                    height=target_size[1],
                ).images

            # 保存图片
            for i, img in enumerate(images):
                save_path = output_dir / f"{image_path.stem}_var{i}.jpg"
                img.save(save_path)
            
            return len(images)

        except Exception as e:
            print(f"❌ 处理出错 {image_path.name}: {e}")
            return 0

def main():
    parser = argparse.ArgumentParser(description="基于 IP-Adapter 的图像变分增强")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="原始图片目录")
    parser.add_argument("--output_dir", type=str, default="data/augmented/ip_adapter_var", help="输出目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="使用的 GPU ID")
    parser.add_argument("--num_vars", type=int, default=2, help="每张原图生成的变体数量")
    parser.add_argument("--scale", type=float, default=0.6, help="IP-Adapter Scale (0.0-1.0), 越高越像原图")
    args = parser.parse_args()

    # 初始化增强器
    augmentor = IPAdapterAugmentor(gpu_id=args.gpu_id, adapter_scale=args.scale)

    raw_path = Path(args.raw_dir)
    output_path = Path(args.output_dir)

    # 递归查找所有图片
    all_images = list(raw_path.rglob("*.jpg"))
    print(f"🔍 扫描到 {len(all_images)} 张原始图片")

    total_generated = 0
    pbar = tqdm(all_images, desc=f"IP-Adapter Variation on GPU {args.gpu_id}")
    
    for img_file in pbar:
        # 准备该类别的输出目录
        rel_path = img_file.relative_to(raw_path)
        class_output_dir = output_path / rel_path.parent
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # 执行增强
        count = augmentor.augment_image(
            img_file, 
            class_output_dir, 
            num_vars=args.num_vars
        )
        total_generated += count
        
        pbar.set_postfix({"New Images": total_generated})

    print(f"🎉 任务完成! 总共生成: {total_generated} 张变体图片")

if __name__ == "__main__":
    main()