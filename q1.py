# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import random
from pathlib import Path

plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 14

# Define data path
DATA = Path('.')
print('OpenCV version:', cv2.__version__)
print('NumPy version: ', np.__version__)

# Load data from lines.csv
D = np.genfromtxt(DATA / 'lines.csv', delimiter=',', skip_header=1)
print('Dataset shape:', D.shape)
print('First 5 rows:\n', D[:5])

# Q1a: Total Least Squares via SVD
x1, y1 = D[:, 0], D[:, 3]

# Mean-center the data
mx, my = np.mean(x1), np.mean(y1)
A = np.column_stack([x1 - mx, y1 - my])

# Perform SVD
_, S, Vt = np.linalg.svd(A)
print('Singular values:', S)

# Extract normal vector (smallest singular value)
a, b = Vt[-1]
c = a * mx + b * my

# Calculate slope and intercept
slope     = -a / b
intercept =  c / b

print(f'\nNormal form:      {a:.6f}·x + {b:.6f}·y = {c:.6f}')
print(f'Slope-intercept:  y = {slope:.6f}·x + {intercept:.6f}')

# Plot data and fitted TLS line
x_range = np.linspace(x1.min(), x1.max(), 300)
y_fit   = slope * x_range + intercept

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x1, y1, s=12, alpha=0.6, color='blue', label='Data (Line 1)')
ax.plot(x_range, y_fit, 'r-', linewidth=2,
        label=f'TLS fit: y = {slope:.3f}x + {intercept:.3f}')
ax.set_xlabel('x₁'); ax.set_ylabel('y₁')
ax.set_title('Q1(a): Total Least Squares — Line 1')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('q1a_tls.png', dpi=150)
plt.show()
print('Result: y =', round(slope,4), '·x +', round(intercept,4))

# Combine all x and y columns into a single dataset
X_cols = D[:, :3]
Y_cols = D[:, 3:]
X_all  = X_cols.flatten()
Y_all  = Y_cols.flatten()
points = np.column_stack([X_all, Y_all])
N = len(points)
print(f'Total points: {N}')

# Helper functions for RANSAC
def fit_line_tls(pts):
    # Fit line using TLS
    mx, my = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - [mx, my])
    a, b = Vt[-1]
    c = a * mx + b * my
    norm = np.hypot(a, b)
    return a/norm, b/norm, c/norm

def point_line_dist(pts, a, b, c):
    # Calculate perpendicular distance from points to a line
    return np.abs(pts[:, 0]*a + pts[:, 1]*b - c)

def ransac_line(pts, n_iter=2000, thresh=0.5, seed=42):
    # RANSAC algorithm for line fitting
    rng = random.Random(seed)
    best_inliers, best_count = None, 0
    for _ in range(n_iter):
        idx = rng.sample(range(len(pts)), 2)
        if np.allclose(pts[idx[0]], pts[idx[1]]):
            continue
        a, b, c = fit_line_tls(pts[idx])
        inliers = np.where(point_line_dist(pts, a, b, c) < thresh)[0]
        if len(inliers) > best_count:
            best_count, best_inliers = len(inliers), inliers
    # Refit line using all inliers
    a, b, c = fit_line_tls(pts[best_inliers])
    return a, b, c, best_inliers

# Extract three lines using RANSAC iteratively
remaining = np.arange(N)
lines     = []
COLORS    = ['orange', 'purple', 'darkgreen']

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(X_all, Y_all, s=6, color='lightgrey', zorder=1, label='All points')

for i in range(3):
    pts = points[remaining]
    a, b, c, inl_idx = ransac_line(pts, thresh=0.5)
    a, b, c = fit_line_tls(pts[inl_idx])   # final refit
    lines.append((a, b, c))

    inl_pts = points[remaining[inl_idx]]
    ax.scatter(inl_pts[:, 0], inl_pts[:, 1], s=10, color=COLORS[i],
               alpha=0.7, zorder=2, label=f'Line {i+1} inliers ({len(inl_idx)})')

    xs = np.linspace(inl_pts[:, 0].min(), inl_pts[:, 0].max(), 300)
    ys = (c - a*xs) / b if abs(b) > 1e-8 else np.full_like(xs, c/a)
    ax.plot(xs, ys, color=COLORS[i], linewidth=2)

    slope_i = -a/b
    intercept_i = c/b
    print(f'Line {i+1}: y = {slope_i:.4f}·x + {intercept_i:.4f}  '
          f'| inliers: {len(inl_idx)}')

    remaining = np.setdiff1d(remaining, remaining[inl_idx])

ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('Q1(b): RANSAC — Three Lines from Combined Data')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('q1b_ransac.png', dpi=150)
plt.show()