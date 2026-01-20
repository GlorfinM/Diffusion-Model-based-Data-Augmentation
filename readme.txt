Project_Root/
│
├── data/
│   ├── raw/                      # [自动下载] 存放 Oxford-IIIT Pet 原始图片 (.jpg)
│   ├── masks/                    # [自动生成] 存放 rembg 生成的二值遮罩 (.png)
│   └── augmented/                # [最终产出] 存放生成的增强数据
│       ├── inpainting_bg/        # 方案A产出：背景替换后的图片
│       └── ip_adapter_var/       # 方案B产出：IP-Adapter 变体图片
│
├── src/
│   ├── preprocess/               # [预处理模块]
│   │   ├── __init__.py
│   │   ├── setup_data.py         # [脚本1] 自动下载并解压数据集
│   │   └── segment.py            # [脚本2] 调用 rembg 批量生成 Mask
│   │
│   └── methods/                  # [数据增强增强算法模块]
│       ├── __init__.py
│       ├── method_inpainting.py  # [脚本3] 方案A：基于 Mask + Text 的背景重绘
│       └── method_ip_adapter.py  # [脚本4] 方案B：基于 Image Prompt 的变体生成
│
├── run_augment.py                # [主入口] 统一命令行接口，调度以上模块
└── readme.txt                     # 项目说明

运行命令示例：
python run_augment.py --step all --gpu_id 1
python run_augment.py --step setup --gpu_id 1
python run_augment.py --step segment --gpu_id 1
python run_augment.py --step ip_adapter --gpu_id 1
python run_augment.py --step inpaint --gpu_id 1

-----------------------------------------------------------
数据增强方案说明
-----------------------------------------------------------

本项目实现了两种基于 Diffusion Model 的数据增强方案：

1. 方案A：基于 Inpainting 的背景替换 (method_inpainting.py)
   - 原理：利用 Stable Diffusion Inpainting 模型，结合分割生成的 Mask，保持前景（宠物）不变，重绘背景。
   - 场景：预设了 5 种不同风格的背景 Prompt（雪地、海滩、丛林、城市、日落）。
   - 目的：增加背景的多样性，提高模型对不同背景的泛化能力。
   - 模型：runwayml/stable-diffusion-inpainting

2. 方案B：基于 IP-Adapter 的变体生成 (method_ip_adapter.py)
   - 原理：使用 IP-Adapter (Image Prompt Adapter) 技术，将原始图片作为视觉提示词输入给 Stable Diffusion。
   - 配置：设置 Adapter Scale 为 0.6，在保持原图主要特征（姿态、颜色、品种）的同时，引入合理的随机变化。
   - 目的：生成与原图相似但细节不同的变体，扩充数据集规模，增加类内多样性。
   - 模型：runwayml/stable-diffusion-v1-5 + h94/IP-Adapter
