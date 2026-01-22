```
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
```

运行命令示例：
```
python run_augment.py --step all --gpu_id 1
python run_augment.py --step setup --gpu_id 1
python run_augment.py --step segment --gpu_id 1
python run_augment.py --step ip_adapter --gpu_id 1
python run_augment.py --step inpaint --gpu_id 1
```

-----------------------------------------------------------
数据增强方案详细说明
-----------------------------------------------------------

本项目旨在利用最前沿的扩散模型（Diffusion Models）技术对宠物数据集进行高质量扩充。相比传统的几何变换（旋转、裁剪）或色彩变换，生成式增强能提供更丰富的语义多样性。

### 方案 A：基于 Inpainting 的背景替换 (Background Replacement via Inpainting)
**核心脚本**: `src/methods/method_inpainting.py`

*   **技术原理**:
    该方案利用 `Stable Diffusion Inpainting` 模型强大的局部重绘能力。它通过输入一张原始图片和一个二值掩膜（Mask），模型会保持 Mask 中白色区域（前景/宠物）的像素完全不变，仅重新生成黑色区域（背景）的内容。生成的背景内容由文本提示词（Text Prompt）精确控制。

*   **处理流程**:
    1.  **前景分割**: 使用 `U2Net` (通过 `rembg` 库) 自动将原始图片中的宠物从背景中剥离，生成高精度的二值 Mask。
    2.  **场景构建**: 预定义了 5 种差异化极大的场景 Prompt（雪山冬日、热带海滩、原始丛林、繁华都市、唯美日落），以确保生成的背景具有高度多样性。
    3.  **定向生成**: 将原图、Mask 和对应场景的 Prompt 输入 Inpainting Pipeline，通过去噪过程生成与前景光影和谐融合的新背景。

*   **方案优势**:
    *   **语义解耦**: 彻底改变图像背景上下文，强迫下游分类模型关注宠物本身的特征，而非依赖背景相关性（例如防止模型错误地认为“草地上的都是狗”）。
    *   **零样本迁移**: 无需针对特定背景训练，仅通过修改 Prompt 即可生成无限种类的环境背景。

### 方案 B：基于 IP-Adapter 的语义变体生成 (Semantic Variation via IP-Adapter)
**核心脚本**: `src/methods/method_ip_adapter.py`

*   **技术原理**:
    该方案采用 `IP-Adapter` (Image Prompt Adapter) 技术。传统的 Stable Diffusion 主要依赖文本提示词生成图像，而 IP-Adapter 引入了一个轻量级的适配器模块，允许模型接受“图片”作为提示词（Image Prompt）。这使得模型能够“看懂”原图的语义特征（如品种、毛色、纹理），并基于此生成新的变体。

*   **处理流程**:
    1.  **特征提取**: 使用图像编码器提取原图的高维视觉特征。
    2.  **混合引导**: 将提取的视觉特征与文本提示词（如 "high quality, realistic"）相结合，共同引导扩散模型的生成过程。
    3.  **强度控制**: 通过调节 `Adapter Scale` (默认 0.6)，在“保持原图身份特征”和“引入随机变化”之间找到最佳平衡点。Scale 越高越像原图，Scale 越低变化越大。

*   **方案优势**:
    *   **姿态与细节重构**: 不同于简单的像素级扰动，IP-Adapter 能生成宠物不同的姿态、表情或微小的毛发细节变化，相当于模拟了对同一只宠物在不同时刻的拍摄。
    *   **保持核心特征**: 相比纯文本生成（Text-to-Image），它能完美保留特定宠物的细粒度特征（如斑点分布、独特的毛色模式），避免生成出“看起来像某种狗但不是这只狗”的错误样本。
