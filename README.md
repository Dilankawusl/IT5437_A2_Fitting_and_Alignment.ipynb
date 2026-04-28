# IT5437 Assignment 2: Fitting and Alignment

## Author

**Name**: M.A.D.P Wijesena  
**Index**: 258850R  
**Course**: IT5437 Computer Vision

## File Structure

- `258850R_IT5437_A2_Fitting_and_Alignment.ipynb`: Main Jupyter notebook with complete implementation and results.
- `lines.csv`: Point data for line fitting.
- `earrings.jpg`: Image for earring size estimation.
- `c1.jpg`, `c2.jpg`: Circuit board images for homography alignment.
- `it5437_assignment_02.pdf`: Assignment specification.
- `README.md`: This file.

## Key Results

### Q1: Line Fitting

* TLS fit: y = 1.2207x - 5.9872
* RANSAC:
  * Line 1: y = -0.4544x + 2.0874 (84 inliers)
  * Line 2: y = 1.2798x - 6.0011 (66 inliers)
  * Line 3: y = 1.0323x + 1.0838 (66 inliers)

### Q2: Earring Size

* Earring 1: 72 px × 78 px → 71.7 mm × 77.8 mm
* Earring 2: 72 px × 78 px → 71.7 mm × 77.8 mm

### Q3: Homography

* Manual (DLT):
  * Mean diff = 18.72 / 255

* SIFT Matching:
  * Keypoints: 1358 vs 1464
  * Good matches: 901

* Auto (SIFT + RANSAC):
  * Mean diff = 17.1 / 255
  * Inliers: 861 / 901

## Requirements

- Python 3.x
- Libraries: NumPy, Matplotlib, OpenCV
- Jupyter Notebook

## How to Run

Open the notebook in Jupyter and run all cells to reproduce the results.

