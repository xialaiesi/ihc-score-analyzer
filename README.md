# IHC Score Analyzer

A desktop application for Immunohistochemistry (IHC) scoring analysis, built with `PyQt5 + OpenCV + scikit-image + matplotlib`. Designed for semi-quantitative analysis of H-DAB / H-E stained images, with support for single-image analysis, ROI selection, batch processing, CSV export, and score overlay visualization.

## Features

- **Color Deconvolution** — Supports `H-DAB`, `H-E`, and `H-DAB (skimage)` stain separation schemes
- **DAB Channel Analysis** — Automatic DAB grayscale extraction with histogram display
- **ROI Selection** — Draw a region of interest for localized scoring
- **H-Score Calculation** — `1 × Low+% + 2 × Pos% + 3 × High+%` (range 0–300)
- **IHC Score** — `Intensity Score × Proportion Score` (range 0–12)
- **Batch Analysis** — Open an entire folder and analyze all images at once with CSV export
- **Visualization** — Original image, DAB channel, Hematoxylin channel, and color-coded score overlay
- **Bilingual UI** — Switch between Chinese and English with one click
- **Unicode Path Support** — Handles Chinese/Unicode file paths on Windows
- **Cross-Platform** — Runs on macOS and Windows; automated builds via GitHub Actions

## Scoring Rules

Based on [IHC Profiler](https://sourceforge.net/projects/ihcprofiler/) standards.

### Grayscale Classification

The DAB channel is classified by grayscale value (lower = deeper staining):

| Category | Grayscale Range | Overlay Color |
|----------|----------------|---------------|
| **High Positive** | 0–60 | Red |
| **Positive** | 61–120 | Orange |
| **Low Positive** | 121–180 | Green |
| **Negative** | 181–235 | Blue |
| **Background** | 236–255 | Excluded |

All thresholds are adjustable via sliders. Three presets are provided: Standard, Strict, and Loose.

### Intensity Score (0–3)

Based on the mean grayscale of all positive pixels:

| Mean Grayscale | Score | Label |
|---------------|-------|-------|
| No positive pixels | 0 | Negative |
| ≤ 110 | 3 | Strong Positive |
| ≤ 150 | 2 | Positive |
| > 150 | 1 | Low Positive |

### Proportion Score (1–4)

Based on the positive rate (High+ + Pos + Low+):

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

| Positive Rate | IHC Score | Result |
|--------------|-----------|--------|
| < 5% | — | **Negative** [-] |
| ≥ 5% | 1–3 | **Positive** [+] |
| ≥ 5% | 4–6 | **Positive** [++] |
| ≥ 5% | 7–12 | **Positive** [+++] |

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
2. Select a stain scheme (H-DAB, H-E, or H-DAB skimage)
3. Adjust thresholds if needed (or use presets)
4. Click **Analyze** for single image, or **Batch Analyze** for the whole folder
5. Optionally select an ROI for localized analysis
6. Export results as CSV or save the score overlay image

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
git tag v1.5.0
git push github v1.5.0
```

## Project Structure

```
ihc_scorer.py               Main application (single-file architecture)
requirements.txt            Python dependencies
docs/changelog.md           Changelog
tools/codex_discuss.py      AI-assisted code discussion tool
.github/workflows/build.yml GitHub Actions CI/CD
```

## License

MIT License
