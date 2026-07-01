# Audio Fingerprinting Using FFT

A Shazam-inspired audio clip locator built in Python for COMM 401 at GUC.

## What It Does
Locates a short audio clip within a longer recording using:
- Sliding-window FFT analysis
- Normalised cosine similarity scoring
- Best-match position detection

## Output
![similarity plot]:
<img width="800" height="372" alt="similarity_plot" src="https://github.com/user-attachments/assets/4b453f1e-5e69-4b8a-93e2-9ce9b0845cbf" />

## Tech Stack
Python · NumPy · SciPy · Matplotlib

## How to Run
1. Install dependencies: `pip install numpy scipy matplotlib`
2. Place your `.wav` files in the project folder
3. Run: `python audio_match.py`

## Author
Ahmed Hagras — GUC Spring 2026
