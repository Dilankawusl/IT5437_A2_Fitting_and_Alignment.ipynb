# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 14

# Define data path
DATA = Path('.')

# Define camera parameters
focal_length_mm = 8.0
pixel_pitch_mm  = 2.2e-3    # 2.2 µm → mm
distance_mm     = 720.0

f_px = focal_length_mm / pixel_pitch_mm   # focal length in pixels
print(f'Focal length : {focal_length_mm} mm  =  {f_px:.1f} px')
print(f'Pixel pitch  : {pixel_pitch_mm*1e3:.1f} µm')
print(f'Distance     : {distance_mm} mm')

# Load and process earring image
img     = cv2.imread(str(DATA / 'earrings.jpg'))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'Image size: {img.shape[1]} × {img.shape[0]} px')

# Apply threshold to isolate earrings
_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find and sort contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours    = sorted([c for c in contours if cv2.contourArea(c) > 500],
                     key=cv2.contourArea, reverse=True)[:2]
print(f'Detected {len(contours)} earring blobs')

# Measure pixel dimensions and convert to real-world size
vis = img_rgb.copy()
print(f'{'Earring':^8} | {'W (px)':^8} | {'H (px)':^8} | {'W (mm)':^8} | {'H (mm)':^8}')
print('-' * 50)

for i, cnt in enumerate(contours):
    x, y, cw, ch = cv2.boundingRect(cnt)
    real_w = cw * pixel_pitch_mm * distance_mm / focal_length_mm
    real_h = ch * pixel_pitch_mm * distance_mm / focal_length_mm
    print(f'{i+1:^8} | {cw:^8} | {ch:^8} | {real_w:^8.1f} | {real_h:^8.1f}')
    cv2.rectangle(vis, (x, y), (x+cw, y+ch), (220, 30, 30), 8)

# Visualize detected earrings
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img_rgb);  axes[0].set_title('Original');                 axes[0].axis('off')
axes[1].imshow(vis);       axes[1].set_title('Detected Earrings (BBox)'); axes[1].axis('off')
plt.suptitle('Q2: Earring Size Estimation', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('q2_earrings.png', dpi=150)
plt.show()
print('\nConclusion: Each earring ≈ 71–72 mm wide × 78 mm tall (~7–8 cm hoop diameter)')