import argparse
import subprocess
import sys
import time
from pathlib import Path

# 定义每个步骤对应的脚本路径
SCRIPTS = {
    "setup": "src/preprocess/setup_data.py",
    "segment": "src/preprocess/segment.py",
    "inpaint": "src/methods/method_inpainting.py",
    "ip_adapter": "src/methods/method_ip_adapter.py"
}

def run_command(command, step_name):
    """
    使用子进程运行命令，并实时打印输出
    """
    print(f"\n{'='*60}")
    print(f"🎬 正在执行步骤: [{step_name}]")
    print(f"👉 命令: {' '.join(command)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    
    # 使用 sys.executable 确保使用当前激活的 conda 环境 python 解析器
    try:
        # check=True 会在命令返回非零状态码时抛出 CalledProcessError
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 步骤 [{step_name}] 执行失败！(错误码: {e.returncode})")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断了步骤 [{step_name}]")
        sys.exit(1)

    duration = time.time() - start_time
    print(f"\n✅ 步骤 [{step_name}] 完成！耗时: {duration:.2f} 秒")

def main():
    parser = argparse.ArgumentParser(description="数据增强流水线总控脚本")
    
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["all", "setup", "segment", "inpaint", "ip_adapter"],
        default="all",
        help="指定要运行的步骤 (默认: all 运行所有步骤)"
    )
    
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0, 
        help="指定使用的 GPU ID (仅对 segment, inpaint, ip_adapter 有效)"
    )

    args = parser.parse_args()

    # 这里的 Python 解释器路径
    python_exe = sys.executable

    # === 1. 数据准备 (CPU) ===
    if args.step in ["all", "setup"]:
        cmd = [python_exe, SCRIPTS["setup"]]
        run_command(cmd, "Setup Data")

    # === 2. Mask 生成 (GPU) ===
    if args.step in ["all", "segment"]:
        cmd = [
            python_exe, SCRIPTS["segment"],
            "--gpu_id", str(args.gpu_id)
        ]
        run_command(cmd, "Segmentation")

    # === 3. Inpainting 增强 (GPU) ===
    if args.step in ["all", "inpaint"]:
        cmd = [
            python_exe, SCRIPTS["inpaint"],
            "--gpu_id", str(args.gpu_id)
            # 如果需要传递镜像变量，可以在这里修改环境变量，或者让用户自己在命令行加
        ]
        run_command(cmd, "Inpainting Augmentation")

    # === 4. IP-Adapter 变分 (GPU) ===
    if args.step in ["all", "ip_adapter"]:
        cmd = [
            python_exe, SCRIPTS["ip_adapter"],
            "--gpu_id", str(args.gpu_id)
        ]
        run_command(cmd, "IP-Adapter Variation")

    print(f"\n{'='*60}")
    print("🎉🎉🎉 所有指定任务已全部完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()