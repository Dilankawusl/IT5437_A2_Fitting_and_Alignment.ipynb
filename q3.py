# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 14

# Define data path
DATA = Path('.')

# Load circuit board images (quarter resolution)
SCALE = 4
im1_full = cv2.imread(str(DATA / 'c1.jpg'))
im2_full = cv2.imread(str(DATA / 'c2.jpg'))
im1 = cv2.resize(im1_full, (0,0), fx=1/SCALE, fy=1/SCALE)
im2 = cv2.resize(im2_full, (0,0), fx=1/SCALE, fy=1/SCALE)
h1, w1 = im1.shape[:2]
h2, w2 = im2.shape[:2]

def to_rgb(bgr): return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

print(f'c1 (quarter-res): {w1}×{h1}')
print(f'c2 (quarter-res): {w2}×{h2}')

# Manual correspondences (quarter-resolution)
# To use on full images, multiply by SCALE.
pts_src = np.float32([
    [219, 222],   # DC jack
    [319, 129],   # USB connector
    [197, 365],   # Top-right corner area
    [326, 341],   # MCU chip top-left
    [234, 478],   # Red reset LED
    [399, 484],   # Bottom-right area
])
pts_dst = np.float32([
    [254, 176],
    [376, 113],
    [195, 308],
    [326, 319],
    [201, 427],
    [359, 476],
])

# Compute homography (no RANSAC — exact 6 pairs)
H_manual, _ = cv2.findHomography(pts_src, pts_dst)
print('Manual Homography H:')
print(H_manual)

# Decode rotation angle from H
cos_t = (H_manual[0,0] + H_manual[1,1]) / 2
angle = np.degrees(np.arccos(np.clip(cos_t, -1, 1)))
print(f'\nApprox rotation: {angle:.1f}°')

# Q3(a): Warp c1 → c2 perspective
warped_manual = cv2.warpPerspective(im1, H_manual, (w2, h2))

# Q3(b): Difference image
diff_manual   = cv2.absdiff(warped_manual, im2)
diff_bright   = cv2.convertScaleAbs(diff_manual, alpha=3.0)
mean_diff_man = diff_manual.mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(to_rgb(im2));           axes[0].set_title('c2 (target)')
axes[1].imshow(to_rgb(warped_manual)); axes[1].set_title('c1 warped → c2 (manual H)')
axes[2].imshow(to_rgb(diff_bright));   axes[2].set_title('|diff| amplified ×3')

# Show correspondence points
for s, d in zip(pts_src, pts_dst):
    axes[0].plot(d[0], d[1], 'g+', markersize=12, markeredgewidth=2)
    axes[1].plot(d[0], d[1], 'g+', markersize=12, markeredgewidth=2)

for ax in axes: ax.axis('off')
plt.suptitle(f'Q3(a/b): Manual Homography (6 correspondences)  |  Mean|diff|={mean_diff_man:.2f}/255',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('q3ab_manual.png', dpi=120)
plt.show()
print(f'Mean |diff| (manual H): {mean_diff_man:.2f} / 255')

# SIFT detection
gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
print(f'Keypoints detected: c1={len(kp1)}  c2={len(kp2)}')

# BFMatcher + Lowe ratio test
bf  = cv2.BFMatcher()
raw = bf.knnMatch(des1, des2, k=2)
good = [m for m, n in raw if m.distance < 0.75 * n.distance]
good_sorted = sorted(good, key=lambda m: m.distance)
print(f'Total raw matches    : {len(raw)}')
print(f'Good (ratio test 0.75): {len(good)}')

# Draw top-80 matches
match_img = cv2.drawMatches(
    im1, kp1, im2, kp2, good_sorted[:80], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig, ax = plt.subplots(figsize=(15, 5))
ax.imshow(to_rgb(match_img))
ax.set_title(f'Q3(c): SIFT Matches — top 80 of {len(good)} (after Lowe ratio test 0.75)',
             fontsize=11, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('q3c_matches.png', dpi=120)
plt.show()

# Auto homography (SIFT + RANSAC)
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H_auto, mask_auto = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
n_inliers = mask_auto.sum()
print(f'RANSAC inliers: {n_inliers}/{len(good)}')
print('\nAuto Homography H:')
print(H_auto)

cos_t_auto = (H_auto[0,0] + H_auto[1,1]) / 2
angle_auto = np.degrees(np.arccos(np.clip(cos_t_auto, -1, 1)))
print(f'\nApprox rotation: {angle_auto:.1f}°')

# Warp & diff
warped_auto = cv2.warpPerspective(im1, H_auto, (w2, h2))
diff_auto   = cv2.absdiff(warped_auto, im2)
diff_auto_b = cv2.convertScaleAbs(diff_auto, alpha=3.0)
mean_diff_auto = diff_auto.mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(to_rgb(im2));          axes[0].set_title('c2 (target)')
axes[1].imshow(to_rgb(warped_auto));  axes[1].set_title('c1 warped → c2 (auto H)')
axes[2].imshow(to_rgb(diff_auto_b));  axes[2].set_title('|diff| amplified ×3')
for ax in axes: ax.axis('off')
plt.suptitle(f'Q3(d): Auto Homography (SIFT+RANSAC, {n_inliers} inliers)  |  '
             f'Mean|diff|={mean_diff_auto:.2f}/255',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('q3d_auto.png', dpi=120)
plt.show()

# Side-by-side comparison of diff images
diff_man_b = cv2.convertScaleAbs(diff_manual, alpha=3.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].imshow(to_rgb(diff_man_b))
axes[0].set_title(f'Manual H  |  Mean|diff|={mean_diff_man:.2f}/255\n(6 hand-picked correspondences)')
axes[1].imshow(to_rgb(diff_auto_b))
axes[1].set_title(f'Auto H  |  Mean|diff|={mean_diff_auto:.2f}/255\n({n_inliers} SIFT+RANSAC inliers)')
for ax in axes: ax.axis('off')
plt.suptitle('Difference Images: Manual vs. Auto Homography (×3 amplified)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('q3_comparison.png', dpi=120)
plt.show()

improvement = (mean_diff_man - mean_diff_auto) / mean_diff_man * 100
print(f'Auto H is {improvement:.1f}% more accurate than manual H')