import os
import numpy as np
import torch, cv2, celldetection as cd
from skimage.data import coins
from matplotlib import pyplot as plt
from PIL import Image
np.random.seed(123)

# Load pretrained model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = cd.fetch_model('ginoro_CpnResNeXt101UNet-fbe875f1a3e5ce2c', check_hash=True).to(device)
model.score_thresh=0.7
model.nms_thresh=0.2
model.eval()

# Load input
img = Image.open('../../_RAW_DATA/122824/Data_122824/Glioma_MDC_2025_training/training0002.jpg')
img = np.array(img.convert('RGB'))

# Run model
with torch.no_grad():
    x = cd.to_tensor(img, transpose=True, device=device, dtype=torch.float32)
    x = x / 255  # ensure 0..1 range
    x = x[None]  # add batch dimension: Tensor[3, h, w] -> Tensor[1, 3, h, w]
    y = model(x)

# Show results for each batch item
contours = y['contours'][0]
scores = y['scores'][0]
print(contours.shape)
print(scores)

fig, ax = plt.subplots(figsize=(10, 10))

# Display the image
im = ax.imshow(img)

# Show masks and boxes on the same axes

contours = [cv2.approxPolyDP(contour.cpu().numpy().astype('int'), epsilon=0.01, closed=True) for contour in contours]
# for contour in contours:
color = np.concatenate([np.random.random(3), np.array([0.0])], axis=0)
h, w = img.shape[:2]
mask = np.ones([h, w, 1])
mask_image =  mask * color.reshape(1, 1, -1)
mask_image = cv2.drawContours(mask_image, contours, -1, (0, 1, 1, 0.5), thickness=2) 

ax.imshow(mask_image)

# Turn off axis labels and ticks (optional, adjust as needed)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Save the figure as an image (replace 'output_image.png' with your desired filename)
fig.savefig("example.png", bbox_inches='tight', pad_inches=0)

plt.close(fig)