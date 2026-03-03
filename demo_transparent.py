import os
import cv2
import torch
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitl', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

image_base = '/home/sgan/transparent/dataset/train_data/images/'
image_array = []
for i in [8,16,32,40,48,58,68,90]:
    idx = f'{image_base}{i:06d}.png'
    image_array.append(idx)

# Create output directory
output_dir = './transparent'
os.makedirs(output_dir, exist_ok=True)

print(f"Processing {len(image_array)} images...")

for img_path in image_array:
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
        
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        print(f"Failed to read image: {img_path}")
        continue
        
    # Perform depth estimation
    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(raw_img[:, :, ::-1])
    plt.title('Raw Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(depth)
    plt.title('Depth Prediction')
    plt.axis('off')
    
    plt.suptitle(f'Prediction with model {encoder}', fontsize=16, fontweight='bold')
    
    # Save output
    base_name = os.path.basename(img_path)
    out_path = os.path.join(output_dir, f'output_{base_name}')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {out_path}")

print(f"All predictions saved to {output_dir}")
