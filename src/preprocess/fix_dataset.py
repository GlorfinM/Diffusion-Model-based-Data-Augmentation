import os
import shutil
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

# === 配置 ===
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
RAW_DIR = Path("data/raw")
TEMP_DIR = Path("data/temp_fix")
TAR_PATH = TEMP_DIR / "images.tar.gz"
EXTRACT_DIR = TEMP_DIR / "images"

def download_file(url, dest_path):
    """流式下载文件并显示进度条"""
    if dest_path.exists():
        print(f"📦 检测到临时文件已存在: {dest_path}，跳过下载。")
        return

    print(f"⬇️  正在下载官方数据集: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1KB

    with open(dest_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def fix_dataset():
    # 1. 准备临时目录
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 下载官方数据集
    try:
        download_file(DATASET_URL, TAR_PATH)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("请检查网络连接。")
        return

    # 3. 解压文件
    print("📦 正在解压数据集 (这可能需要几秒钟)...")
    if not EXTRACT_DIR.exists():
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            # 官方压缩包里自带一个 'images' 文件夹，我们解压到 TEMP_DIR 下
            tar.extractall(path=TEMP_DIR)
    
    # 解压后的路径通常是 data/temp_fix/images
    source_images_dir = TEMP_DIR / "images"
    if not source_images_dir.exists():
        print(f"❌ 解压结构异常，未找到 {source_images_dir}")
        return

    # 4. 遍历并修补
    print(f"🔍 开始对比并修补 data/raw ...")
    
    # 获取所有解压出来的图片
    all_source_files = list(source_images_dir.glob("*.jpg"))
    total_files = len(all_source_files)
    print(f"📄 官方源共包含 {total_files} 张图片。")

    fixed_count = 0
    skipped_count = 0

    for src_file in tqdm(all_source_files, desc="Verifying & Fixing"):
        # 解析文件名以确定它属于哪个类别
        # 格式: Class_Name_Number.jpg (例如 Abyssinian_100.jpg 或 Saint_Bernard_10.jpg)
        filename = src_file.name
        
        # 逻辑: 从右边数第一个下划线切分，左边就是类别名
        # "Abyssinian_100.jpg" -> "Abyssinian"
        # "Saint_Bernard_10.jpg" -> "Saint_Bernard"
        try:
            class_name = filename.rsplit("_", 1)[0]
        except IndexError:
            # 极少数异常文件处理
            continue

        # 构建目前应该存在的路径
        target_dir = RAW_DIR / class_name
        target_file = target_dir / filename

        # 检查是否存在
        if not target_file.exists():
            # 🚨 发现缺失！执行修补
            # 确保目标文件夹存在 (以防万一整个类都缺了)
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            shutil.copy2(src_file, target_file)
            fixed_count += 1
        else:
            skipped_count += 1

    # 5. 总结与清理
    print("\n" + "="*40)
    print(f"🎉 校验修复完成！")
    print(f"   - ✅ 现有匹配: {skipped_count}")
    print(f"   - 🔧 修复缺失: {fixed_count} (这些文件已被补入 data/raw)")
    print("="*40)

    询问是否删除临时文件
    # 为了自动化，这里默认清理，如果你想保留可以注释掉
    print("🧹 正在清理临时下载文件...")
    try:
        if TAR_PATH.exists():
            os.remove(TAR_PATH)
        if source_images_dir.exists():
            shutil.rmtree(source_images_dir)
        # 删除 temp_fix 文件夹本身
        if TEMP_DIR.exists():
            TEMP_DIR.rmdir()
        print("✅ 清理完成。")
    except Exception as e:
        print(f"⚠️ 清理临时文件时出错 (不影响数据集): {e}")

if __name__ == "__main__":
    fix_dataset()