# IT5437 Assignment 2: Fitting and Alignment

## Summary

This repository implements Assignment 2 for IT5437 Computer Vision, covering line fitting, size estimation, and image alignment techniques.

## File Structure

- `258850R_IT5437_A2_Fitting_and_Alignment.ipynb`: Main Jupyter notebook with complete implementation.
- `lines.csv`: Point data for line fitting.
- `earrings.jpg`: Image for size estimation.
- `c1.jpg`, `c2.jpg`: Images for homography.
- `it5437_assignment_02.pdf`: Assignment PDF.
- `README.md`: This file.

## Key Results

### Q1: Line Fitting
- TLS fit: y ≈ -0.12x + 45.7
- RANSAC extracted 3 lines from combined data.

### Q2: Earring Size
- Dimensions: ~72 mm × 78 mm using pinhole camera model.

### Q3: Homography
- Manual (6 points): Mean diff = 18.3/255
- Auto (SIFT+RANSAC): Mean diff = 17.1/255 (~7% better)

## Requirements
- Python 3.x, NumPy, Matplotlib, OpenCV

## Author
M.A.D.P Wijesena (258850R)