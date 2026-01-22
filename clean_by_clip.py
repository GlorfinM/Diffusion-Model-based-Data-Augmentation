import argparse
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import os

# === 配置区域 ===
# 阈值设定：
# 0.8: 非常严格，稍微有一点变样就删 (保留下来的质量极高，但数量少)
# 0.7: 适中 (推荐)
# 0.6: 宽松 (只删除那种完全变成乱码的图)
THRESHOLD = 0.8

class CLIPCleaner:
    def __init__(self, gpu_id=0):
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        print(f"🚀 [GPU {gpu_id}] 初始化 CLIP 看门狗 (openai/clip-vit-base-patch32)...")
        
        # 加载 CLIP 模型
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def calculate_similarity(self, img_path_a, img_path_b):
        try:
            image_a = Image.open(img_path_a).convert("RGB")
            image_b = Image.open(img_path_b).convert("RGB")

            # 预处理
            inputs = self.processor(images=[image_a, image_b], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            # 归一化特征
            features = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = (features[0] @ features[1].T).item()
            return similarity
        
        except Exception as e:
            print(f"⚠️ 读取错误: {e}")
            return 0.0

def main():
    parser = argparse.ArgumentParser(description="使用 CLIP 清洗生成的垃圾数据")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="原始图片目录")
    parser.add_argument("--aug_dir", type=str, default="data/augmented/sdedit_opt", help="生成图片目录")
    parser.add_argument("--delete", action="store_true", help="【危险】加上此参数才会真删，否则只打印")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    cleaner = CLIPCleaner(gpu_id=args.gpu_id)
    
    raw_root = Path(args.raw_dir)
    aug_root = Path(args.aug_dir)
    
    # 查找所有生成的图片
    aug_images = list(aug_root.rglob("*.jpg"))
    print(f"🔍 扫描到 {len(aug_images)} 张增强图片，开始 CLIP 质检 (阈值: {THRESHOLD})...")

    deleted_count = 0
    bad_files = []

    for aug_file in tqdm(aug_images, desc="Auditing"):
        # 1. 找到对应的原图
        # 假设生成图文件名是: Beagle_01_sketch.jpg
        # 原图文件名应该是: Beagle_01.jpg
        # 我们需要去掉后缀 (_sketch, _oil)
        
        # 简单粗暴的方法：尝试移除最后一部分下划线后缀
        stem = aug_file.stem # Beagle_01_sketch
        original_stem = "_".join(stem.split("_")[:-1]) # Beagle_01
        
        # 在 raw 目录下寻找原图 (保持目录结构一致性)
        rel_path = aug_file.parent.relative_to(aug_root) # specific_class/
        raw_file = raw_root / rel_path / f"{original_stem}.jpg"

        if not raw_file.exists():
            # 尝试另一种命名逻辑 (有的文件名本身带下划线)
            # 这里的逻辑需要根据你的实际命名规则微调
            # 比如直接遍历 raw_root 找同名文件可能太慢，最好保持文件夹结构一致
            continue

        # 2. 计算相似度
        score = cleaner.calculate_similarity(raw_file, aug_file)
        
        # 3. 判定
        if score < THRESHOLD:
            bad_files.append((aug_file, score))
            
            if args.delete:
                try:
                    os.remove(aug_file)
                    deleted_count += 1
                except:
                    pass

    print("\n" + "="*50)
    print("📊 清洗报告")
    print("="*50)
    if bad_files:
        print(f"❌ 发现 {len(bad_files)} 张不合格图片 (相似度 < {THRESHOLD}):")
        for i, (fp, score) in enumerate(bad_files[:10]):
            status = "已删除" if args.delete else "建议删除"
            print(f"  {i+1}. [{status}] Score {score:.3f}: {fp.name}")
        if len(bad_files) > 10:
            print(f"  ... 以及其他 {len(bad_files)-10} 张")
            
        if not args.delete:
            print(f"\n💡 请运行: python clean_by_clip.py --delete --aug_dir {args.aug_dir} 来执行删除。")
        else:
            print(f"🗑️ 已成功删除 {deleted_count} 张垃圾图片。")
    else:
        print("✨ 所有图片质量均达标！")

if __name__ == "__main__":
    main()