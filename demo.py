import cv2
import torch
import time
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
    idx = image_base + f'{i:06d}.png'
    image_array.append(idx)

print(image_array)

### 消耗掉初始化网络的时间
t00 = time.perf_counter()
raw_img = cv2.imread('/home/sgan/Depth-Anything-V2/images/new/IMG_2685.png')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
###

t0 = time.perf_counter()
raw_img = cv2.imread('/home/sgan/Depth-Anything-V2/images/new/IMG_2685.png')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy

t1 = time.perf_counter()

raw_img2 = cv2.imread('/home/sgan/Depth-Anything-V2/images/new/IMG_2686.png')
depth2 = model.infer_image(raw_img2) # HxW raw depth map in numpy

t2 = time.perf_counter()

render_time_1 = t1-t0
render_time_2 = t2-t1


plt.figure(figsize=(10, 10))

plt.subplot(2,2,1)
plt.imshow(raw_img[:, :, ::-1])
plt.title('Raw 1')

plt.subplot(2,2,2)
plt.imshow(depth)
plt.title(f'Render 1, elapsed {round(render_time_1,3)}')

plt.subplot(2,2,3)
plt.imshow(raw_img2[:, :, ::-1])
plt.title('Raw 2')

plt.subplot(2,2,4)
plt.imshow(depth2)
plt.title(f'Render 2, elapsed {round(render_time_2,3)}')

# plt.show()

ctrl_number = '2-2-l'

plt.suptitle(f'Prediction with model {encoder}', fontsize=20, fontweight='bold')

plt.savefig(f'./images/output/demo_output_{ctrl_number}.png',dpi=600)