import argparse
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 防止 PIL 遇到大图报错，虽然这里主要是小图，但加上保险
Image.MAX_IMAGE_PIXELS = None

def check_file_integrity(file_path):
    """
    双重检测：
    1. 检查文件大小是否为 0
    2. 检查 PIL 是否能正常打开并验证内容
    返回: (是否完好, 错误原因)
    """
    # 检测 1: 硬性标准 - 0 字节文件
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return False, "空文件 (0 bytes)"
    except OSError:
        return False, "文件无法访问"

    # 检测 2: 软性标准 - 图片头部或数据损坏
    try:
        with Image.open(file_path) as img:
            img.verify()  # 尝试读取文件头和结构，不解码像素，速度快且能发现截断
    except Exception as e:
        return False, f"损坏无法读取 ({str(e)})"

    return True, "OK"

def main():
    parser = argparse.ArgumentParser(description="数据集坏文件清理工具")
    parser.add_argument("--target_dir", type=str, default="data/augmented", help="要扫描的根目录")
    parser.add_argument("--delete", action="store_true", help="【危险】添加此参数才会真正删除文件，否则仅扫描")
    args = parser.parse_args()

    root_path = Path(args.target_dir)
    if not root_path.exists():
        print(f"❌ 目录不存在: {root_path}")
        return

    print(f"🔍 正在扫描目录: {root_path}")
    if args.delete:
        print("⚠️  [警告] 正在运行【删除模式】！坏文件将被物理删除！")
    else:
        print("🛡️  [提示] 正在运行【演习模式】。不会删除任何文件。")

    # 递归查找所有 .jpg / .png 文件
    # 如果你的后缀不只是 jpg，可以在这里添加
    extensions = ['*.jpg', '*.jpeg', '*.png']
    all_files = []
    for ext in extensions:
        all_files.extend(list(root_path.rglob(ext)))
    
    print(f"📄 找到 {len(all_files)} 个图片文件，开始完整性检查...")

    bad_files = []
    
    # 使用 tqdm 显示进度
    for file_path in tqdm(all_files, desc="Checking integrity"):
        is_valid, reason = check_file_integrity(file_path)
        
        if not is_valid:
            bad_files.append((file_path, reason))
            # 如果是删除模式，直接删
            if args.delete:
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"\n❌ 删除失败 {file_path}: {e}")

    # === 输出总结报告 ===
    print("\n" + "="*50)
    print(f"📊 扫描报告 - {root_path}")
    print("="*50)
    print(f"✅ 完好文件: {len(all_files) - len(bad_files)}")
    print(f"❌ 损坏文件: {len(bad_files)}")
    
    if len(bad_files) > 0:
        print("\n[损坏文件详情]")
        # 只打印前 10 个，避免刷屏
        for i, (fp, reason) in enumerate(bad_files):
            status = "已删除" if args.delete else "未删除"
            print(f"  {i+1}. [{status}] {reason}: {fp.name}")
            if i >= 9:
                print(f"  ... 以及其他 {len(bad_files)-10} 个文件")
                break
        
        print("-" * 50)
        if not args.delete:
            print(f"💡 发现 {len(bad_files)} 个坏文件。请添加 --delete 参数再次运行以清理它们。")
            print(f"   命令示例: python clean_dataset.py --target_dir {args.target_dir} --delete")
        else:
            print(f"🗑️  成功清理 {len(bad_files)} 个坏文件。")
            print("🚀 现在你可以重新运行生成脚本，它们会自动填补这些空缺。")
    else:
        print("✨ 完美！没有发现损坏的文件。")

if __name__ == "__main__":
    main()