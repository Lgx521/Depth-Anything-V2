import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# --- 模型配置 (从官方代码复制) ---
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

# --- 加载模型 ---
encoder = 'vitb' # 你可以选择 'vits', 'vitb', 或 'vitl'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()
print(f"模型 {encoder} 已加载到 {DEVICE}")

# --- 读取你的 RealSense 图像 ---
# 在这里填入你的 RealSense 彩色图像路径
# image_path = '/home/sgan/mirror_awareness/general_test/output/low_resolution/20251015_210412_color.png'
image_path = '/home/sgan/mirror_awareness/util_scripts/output/window.png'
raw_img = cv2.imread(image_path)

if raw_img is None:
    print(f"错误: 无法读取图像 {image_path}")
else:
    # --- 推理 ---
    # 你可以在这里指定输入尺寸，与 --input-size 参数效果相同
    # 如果不指定，默认使用 518
    # depth = model.infer_image(raw_img, input_size=1024)
    depth = model.infer_image(raw_img, input_size=518)
    print("深度图推理完成")

    # --- 可视化和保存结果 ---
    # 使用 matplotlib 显示结果
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # axes[0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    # axes[0].set_title('原始 RealSense 图像')
    # axes[0].axis('off')
    
    # # Depth Anything V2 输出的是相对深度，使用 inferno colormap 效果较好
    # axes[1].imshow(depth, cmap='inferno')
    # axes[1].set_title('生成的深度图')
    # axes[1].axis('off')
    
    # plt.tight_layout()
    # plt.savefig('realsense_depth_result.png')
    # plt.show()

    # 如果需要保存为灰度图（原始深度值）
    # 注意：这只是相对深度，不是以米为单位的真实深度
    # 归一化到 0-255 的 8-bit 图像以便保存
    depth_normalized = cv2.normalize(depth, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # depth_gray = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR) # 转为3通道以应用colormap
    # depth_colored = cv2.applyColorMap(depth_gray, cv2.COLORMAP_INFERNO)
    # cv2.imwrite('realsense_depth_colored.png', depth_colored)
    cv2.imwrite('realsense_depth_gray.png', depth_normalized)
    print("结果已保存为 realsense_depth_result.png 和 realsense_depth_colored.png")