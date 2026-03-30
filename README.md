# IHC Score Analyzer

A desktop application for Immunohistochemistry (IHC) scoring analysis, built with `PyQt5 + OpenCV + Pillow + matplotlib`. Designed for semi-quantitative analysis of IHC stained images, with support for single-image analysis, ROI selection, batch processing, CSV export, and score overlay visualization.

## Features

- **CLAHE Preprocessing** — Gaussian blur + CLAHE (LAB L-channel) for contrast enhancement
- **HSV Positive Detection** — Automatic detection of brown (DAB) positive regions via HSV color space (Hue 0–20, Saturation ≥ 50, Value ≥ 50) with morphological refinement
- **Intensity Classification** — Masked-image grayscale analysis for 4-level scoring
- **ROI Selection** — Draw a region of interest for localized scoring
- **H-Score Calculation** — `1 × Low+% + 2 × Pos% + 3 × High+%` (range 0–300)
- **IHC Score** — `Intensity Score × Proportion Score` (range 0–12)
- **Batch Analysis** — Open a folder, analyze all images at once, results cached for instant browsing
- **Image Navigation** — Browse images with toolbar arrows, side-panel arrows, or click table rows
- **Visualization** — Original image, positive region (HSV masked), preprocessed image, and color-coded score overlay
- **Bilingual UI** — Switch between Chinese and English with one click (Times New Roman for English)
- **TIFF Support** — Enhanced TIFF loading via Pillow with cv2 fallback; handles Unicode file paths
- **Cross-Platform** — Runs on macOS and Windows; automated builds via GitHub Actions

## Scoring Rules

### Analysis Pipeline

1. **Preprocessing** — GaussianBlur(3×3) + CLAHE(clipLimit=3.0, tileGridSize=8×8) on LAB L-channel
2. **HSV Detection** — Extract brown/DAB regions using HSV thresholds, followed by morphological close + open operations
3. **Intensity Classification** — Classify pixels in the masked image by grayscale value

### Grayscale Classification

The masked image (positive regions only) is classified by grayscale value (higher = stronger staining):

| Category | Grayscale Range | Overlay Color |
|----------|----------------|---------------|
| **High Positive** | ≥ 160 | Red |
| **Positive** | 100–159 | Orange |
| **Low Positive** | 40–99 | Green |
| **Negative** | < 40 | Blue |

### Intensity Score (0–3)

Based on the mean grayscale of positive pixels (gray > 0 in masked image):

| Mean Grayscale | Score | Label |
|---------------|-------|-------|
| < 40 | 0 | Negative |
| 40–99 | 1 | Low Positive |
| 100–159 | 2 | Positive |
| ≥ 160 | 3 | Strong Positive |

### Proportion Score (1–4)

Based on the HSV-detected positive pixel ratio:

| Positive Rate | Score |
|--------------|-------|
| 0% – 25% | 1 |
| 25% – 50% | 2 |
| 50% – 75% | 3 |
| 75% – 100% | 4 |

### IHC Score

```
IHC Score = Intensity Score × Proportion Score
```

Range: **0–12**

### Clinical Determination

| Total Positive % | IHC Score | Result |
|-----------------|-----------|--------|
| ≤ 5% | — | **Negative** [-] |
| > 5% | 1–3 | **Positive** [+] |
| > 5% | 4–6 | **Positive** [++] |
| > 5% | 7–12 | **Positive** [+++] |

## Batch Export CSV Columns

| Column | Description |
|--------|-------------|
| No. | Row index |
| Filename | Image file name |
| Total Pixels | W × H of the image |
| High+(%) | High Positive percentage |
| Pos(%) | Positive percentage |
| Low+(%) | Low Positive percentage |
| Neg(%) | Negative percentage |
| Clinical | Positive / Negative |
| Intensity | Intensity Score (0–3) |
| Proportion | Proportion Score (1–4) |
| IHC Score | Final score (0–12) |

## Installation

### Option 1: Download Pre-built Binaries

Go to [Releases](https://github.com/xialaiesi/ihc-score-analyzer/releases):

- **Windows** — `IHC_Score_Analyzer.exe` (double-click to run)
- **macOS** — `IHC_Score_Analyzer_macOS.zip` (unzip, then open the `.app`)
  - If blocked: right-click → Open

### Option 2: Run from Source

```bash
git clone https://github.com/xialaiesi/ihc-score-analyzer.git
cd ihc-score-analyzer

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 ihc_scorer.py
```

### Requirements

- Python 3.9+
- PyQt5, OpenCV, NumPy, scikit-image, Matplotlib, Pillow

## Usage

1. Click **Open Image** or **Open Folder** to load IHC stained images
2. Click **Analyze** for single image, or **Batch Analyze** for the whole folder
3. Use **◀ ▶ arrows** (toolbar or image sides) to browse images — analysis results switch automatically
4. Click any **table row** to jump to that image
5. Optionally select an **ROI** for localized analysis
6. **Export** results as CSV or save the score overlay image

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open Image |
| `Ctrl+D` | Open Folder |
| `Ctrl+E` | Export CSV |
| `Ctrl+S` | Save Analysis Image |
| Scroll Wheel | Zoom Image |

## Building

```bash
pip install pyinstaller
python3 -c "import py_compile; py_compile.compile('ihc_scorer.py', doraise=True)"
pyinstaller --onefile --windowed --name "IHC_Score_Analyzer" --noconfirm ihc_scorer.py
```

Cross-platform builds are automated via GitHub Actions (`.github/workflows/build.yml`). Push a version tag to trigger:

```bash
git tag v1.7.0
git push github v1.7.0
```

## Project Structure

```
ihc_scorer.py               Main application (single-file architecture)
requirements.txt            Python dependencies
docs/changelog.md           Changelog
.github/workflows/build.yml GitHub Actions CI/CD
```

## License

MIT License
