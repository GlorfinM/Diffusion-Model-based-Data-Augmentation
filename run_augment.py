import argparse
import subprocess
import sys
import time
from pathlib import Path

# Define script paths
SCRIPTS = {
    "setup": "src/preprocess/setup_data.py",
    "segment": "src/preprocess/segment.py",
    "inpaint": "src/methods/method_inpainting.py",
    "ip_adapter": "src/methods/method_ip_adapter.py"
}

def run_command(command, step_name):
    """
    Run command as subprocess and stream output
    """
    print(f"\n{'='*60}")
    print(f"Executing step: [{step_name}]")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}\n")

    start_time = time.time()
    
    # Use sys.executable to ensure current conda environment is used
    try:
        # check=True raises CalledProcessError on non-zero exit code
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nStep [{step_name}] failed! (Exit code: {e.returncode})")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\nUser interrupted step [{step_name}]")
        sys.exit(1)

    duration = time.time() - start_time
    print(f"\nStep [{step_name}] completed! Duration: {duration:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Data Augmentation Pipeline Controller")
    
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["all", "setup", "segment", "inpaint", "ip_adapter"],
        default="all",
        help="Step to run (default: all)"
    )
    
    parser.add_argument(
        "--gpu_id", 
        type=int, 
        default=0, 
        help="GPU ID (for segment, inpaint, ip_adapter)"
    )

    args = parser.parse_args()

    # Current Python executable
    python_exe = sys.executable

    # === 1. Setup Data (CPU) ===
    if args.step in ["all", "setup"]:
        cmd = [python_exe, SCRIPTS["setup"]]
        run_command(cmd, "Setup Data")

    # === 2. Segmentation (GPU) ===
    if args.step in ["all", "segment"]:
        cmd = [
            python_exe, SCRIPTS["segment"],
            "--gpu_id", str(args.gpu_id)
        ]
        run_command(cmd, "Segmentation")

    # === 3. Inpainting (GPU) ===
    if args.step in ["all", "inpaint"]:
        cmd = [
            python_exe, SCRIPTS["inpaint"],
            "--gpu_id", str(args.gpu_id)
        ]
        run_command(cmd, "Inpainting Augmentation")

    # === 4. IP-Adapter (GPU) ===
    if args.step in ["all", "ip_adapter"]:
        cmd = [
            python_exe, SCRIPTS["ip_adapter"],
            "--gpu_id", str(args.gpu_id)
        ]
        run_command(cmd, "IP-Adapter Variation")

    print(f"\n{'='*60}")
    print("Pipeline finished successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()