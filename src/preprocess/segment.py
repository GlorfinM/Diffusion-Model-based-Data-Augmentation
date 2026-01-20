import os
import argparse
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile

# 防止部分图片因截断报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser(description="使用 Rembg (U2Net) 批量生成宠物图片的背景遮罩")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="data/raw", 
        help="原始图片根目录 (包含类别子文件夹)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/masks", 
        help="输出遮罩根目录 (将自动创建对应的子文件夹)"
    )
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0, 
        help="指定使用的 GPU ID (例如 0 或 1)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="u2net", 
        choices=["u2net", "u2netp", "u2net_human_seg"],
        help="使用的 Rembg 模型 (u2net 精度最高)"
    )
    
    return parser.parse_args()

def process_segmentation(args):
    # ---------------------------------------------------------
    # 1. 环境设置 (在导入 rembg/onnxruntime 之前设置 GPU)
    # ---------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"⚙️  已指定使用 GPU: {args.gpu_id}")

    # 延迟导入 rembg，确保环境变量先生效
    try:
        from rembg import new_session, remove
    except ImportError:
        print("❌ 错误: 未找到 rembg 库。请运行: pip install 'rembg[gpu]'")
        return

    # 路径封装
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if not input_root.exists():
        print(f"❌ 错误: 输入目录不存在: {input_root}")
        return

    # ---------------------------------------------------------
    # 2. 扫描文件
    # ---------------------------------------------------------
    print(f"🔍 正在扫描 {input_root} 下的图片...")
    # 递归查找所有 jpg/png
    extensions = ['*.jpg', '*.jpeg', '*.png']
    all_files = []
    for ext in extensions:
        all_files.extend(list(input_root.rglob(ext)))
    
    if not all_files:
        print("⚠️  未找到任何图片文件。")
        return
        
    print(f"✅ 找到 {len(all_files)} 张图片，准备处理...")

    # ---------------------------------------------------------
    # 3. 初始化模型 Session
    # ---------------------------------------------------------
    print(f"🚀 正在 GPU 上加载模型 '{args.model}' ...")
    start_time = time.time()
    
    # 显式指定 CUDA Provider，虽然设置了 CUDA_VISIBLE_DEVICES，
    # 但显式指定能确保 onnxruntime 不会回退到 CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = new_session(model_name=args.model, providers=providers)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("提示: 请检查是否安装了 onnxruntime-gpu")
        return

    # ---------------------------------------------------------
    # 4. 批量处理循环
    # ---------------------------------------------------------
    success_count = 0
    skip_count = 0
    error_count = 0

    # 使用 tqdm 显示进度条
    pbar = tqdm(all_files, desc="Processing Masks", unit="img")
    
    for img_path in pbar:
        try:
            # 构建相对路径 (例如: Abyssinian/Abyssinian_1.jpg)
            rel_path = img_path.relative_to(input_root)
            
            # 构建输出路径 (data/masks/Abyssinian/Abyssinian_1.png)
            # 注意: Mask 统一存为 png 格式以保持无损
            out_path = output_root / rel_path.with_suffix('.png')
            
            # --- 断点续传检查 ---
            if out_path.exists():
                skip_count += 1
                continue
            
            # 确保父目录存在
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # --- 核心分割逻辑 ---
            # 读取图片
            with open(img_path, 'rb') as i:
                input_data = i.read()
                
            # 推理 (Running Inference)
            # only_mask=True: 返回黑白 Mask (白前景，黑背景)
            output_data = remove(input_data, session=session, only_mask=True)
            
            # 保存结果
            with open(out_path, 'wb') as o:
                o.write(output_data)
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            # 在进度条旁打印简短错误，不打断整体进度
            pbar.write(f"⚠️  Error processing {img_path.name}: {str(e)}")

    # ---------------------------------------------------------
    # 5. 总结
    # ---------------------------------------------------------
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*40)
    print(f"🎉 处理完成！耗时: {duration:.2f} 秒")
    print(f"   - ✅ 成功生成: {success_count}")
    print(f"   - ⏭️  跳过已存在: {skip_count}")
    print(f"   - ❌ 失败: {error_count}")
    print(f"   - 📂 结果保存在: {output_root}")
    print("="*40)

if __name__ == "__main__":
    args = parse_args()
    process_segmentation(args)
