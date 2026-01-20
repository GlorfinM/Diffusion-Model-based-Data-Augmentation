import os
import requests
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

# === 配置区域 ===
# Oxford-IIIT Pet 数据集官方下载链接
DATA_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"

# 路径定义
# 最终结构: data/raw/<class_name>/<image_name>.jpg
ROOT_DIR = Path("data")
RAW_DIR = ROOT_DIR / "raw"

def download_file(url, destination):
    """流式下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB

    print(f"⬇️  正在下载数据集: {url}")
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
    将混杂在 source_dir 下的所有图片，根据文件名按类别整理到 RAW_DIR 的子文件夹中。
    文件名格式示例: Abyssinian_1.jpg -> 类别: Abyssinian
    """
    print("🗂️  正在按类别重组文件结构...")
    
    # 获取所有 jpg 图片
    images = list(source_dir.glob("*.jpg"))
    if not images:
        print(f"⚠️  在 {source_dir} 中未找到图片。")
        return

    for img_path in tqdm(images, desc="Organizing"):
        # 解析文件名获取类别 (例如: "Abyssinian_100.jpg" -> "Abyssinian")
        # 逻辑：取最后一个下划线之前的所有字符作为类别名
        filename = img_path.name
        if "_" in filename:
            class_name = "_".join(filename.split("_")[:-1])
        else:
            # 异常文件名处理 (虽然数据集中通常没有)
            class_name = "Uncategorized"
        
        # 目标文件夹: data/raw/<class_name>
        target_dir = RAW_DIR / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动文件
        shutil.move(str(img_path), str(target_dir / filename))

    # 删除已被掏空的原始 source_dir (即 data/raw/images)
    try:
        shutil.rmtree(source_dir)
        print(f"🧹 已清理临时目录: {source_dir}")
    except OSError as e:
        print(f"⚠️  无法删除临时目录 {source_dir}: {e}")

def setup_oxford_pet_dataset():
    """主函数"""
    
    # 1. 检查数据是否似乎已经准备好了
    # 如果 data/raw 下已经有子文件夹（不包括 tar.gz），则认为已完成
    existing_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()] if RAW_DIR.exists() else []
    if len(existing_dirs) > 10: # 简单的启发式检查，如果子文件夹超过10个，说明已经整理过了
        print(f"✅ 数据集似乎已准备就绪 (检测到 {len(existing_dirs)} 个类别文件夹)。")
        return

    # 确保 data/raw 存在
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = RAW_DIR / "images.tar.gz"

    # 2. 下载 (如果压缩包不存在)
    if not tar_path.exists():
        try:
            download_file(DATA_URL, tar_path)
        except KeyboardInterrupt:
            print("\n❌ 下载中断，清理未完成文件...")
            if tar_path.exists(): os.remove(tar_path)
            return
        except Exception as e:
            print(f"\n❌ 下载出错: {e}")
            return
    else:
        print(f"📦 检测到压缩包已存在，跳过下载。")

    # 3. 解压
    # 官方包解压后会得到一个名为 "images" 的文件夹
    extract_temp_dir = RAW_DIR / "images" 
    
    # 如果之前解压过一部分但没整理，先清理掉避免冲突
    if extract_temp_dir.exists():
        shutil.rmtree(extract_temp_dir)

    print("📦 正在解压...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=RAW_DIR)
    except tarfile.TarError as e:
        print(f"❌ 解压失败: {e}")
        return

    # 4. 整理目录结构 (Re-organize)
    if extract_temp_dir.exists():
        organize_dataset(extract_temp_dir)
    else:
        print("❌ 错误：解压后未找到预期的 'images' 文件夹，请检查数据集源文件结构。")
        return

    # 5. 清理压缩包
    print("🧹 正在清理压缩包...")
    os.remove(tar_path)
    
    # 6. 最终统计
    class_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    total_images = sum([len(list(d.glob("*.jpg"))) for d in class_dirs])
    print(f"🎉 全部完成！")
    print(f"   - 类别数量: {len(class_dirs)}")
    print(f"   - 图片总数: {total_images}")
    print(f"   - 存储路径: {RAW_DIR}")

if __name__ == "__main__":
    setup_oxford_pet_dataset()