# Video Duplicate Finder

## Overview

An advanced Python tool that finds duplicate video files in a specified directory using multiple sophisticated comparison methods including perceptual hashing, color histogram analysis, audio and temporal fingerprints and metadata comparison for highly accurate duplicate detection.

It can be used in three ways:
- **Terminal UI** (`make tui`) - scan, browse results, play files and move duplicates to the Trash without leaving the terminal
- **Command line** (`make scan DIR=...`) - non-interactive, scriptable, optional JSON output
- **Interactive prompts** (`make run`) - the original question-and-answer mode

## Features

### 🔍 **Advanced Detection Methods**
- **Multiple Hash Algorithms**: Perceptual Hash (pHash), Difference Hash (dHash), Average Hash, and Combined methods
- **Color Histogram Analysis**: Compares color distribution patterns across video frames
- **Smart Frame Sampling**: Analyzes frames from different parts of the video (not just the beginning)
- **Multi-factor Similarity Scoring**: Combines hash similarity, duration, file size, FPS, and color patterns

### 📊 **Comprehensive Analysis**
- **Video Metadata Extraction**: Duration, FPS, total frames, file size
- **Intelligent Similarity Threshold**: Configurable sensitivity (0.0-1.0)
- **Enhanced File Format Support**: .mp4, .avi, .mkv, .mov, .m4v, .wmv, .flv, .webm, .mts, .m2ts
- **Detailed Reporting**: Shows similarity scores, file sizes, durations, and recommendations

### 💡 **Smart Recommendations**
- **File Retention Suggestions**: Recommends which duplicate to keep based on quality/size
- **Space Savings Calculator**: Shows potential storage savings
- **Error Handling**: Gracefully handles corrupted or unreadable video files
- **Progress Logging**: Colored console output with detailed analysis progress

### 🖥 **Terminal UI & easy setup**
- **One-command setup**: `make install` creates a virtualenv and installs everything (uses [uv](https://docs.astral.sh/uv/) when available)
- **Full-screen TUI**: settings form, live progress bar, results table with file details and a live scan log
- **Act on results**: play either file, reveal it in Finder, move the unwanted copy to the Trash
- **JSON export**: save results (with the keep suggestion) from the TUI (`e`) or CLI (`--json`)

## Quick Start

Requires Python 3.8+ and (optionally, for audio fingerprinting) `ffmpeg` (`brew install ffmpeg`).

```bash
cd video_duplicate_finder
make install          # creates .venv and installs requirements.txt (a few minutes the first time)
make tui              # terminal UI
```

All `make` targets (`make help` lists them):

| Command | What it does |
| --- | --- |
| `make install` | Create `.venv` and install dependencies (re-runs automatically when `requirements.txt` changes) |
| `make tui [DIR=~/Movies]` | Launch the terminal UI, optionally with the directory pre-filled |
| `make run` | Classic interactive CLI (asks for every option) |
| `make scan DIR=~/Movies` | Non-interactive scan with defaults |
| `make check` | Verify packages and `ffmpeg` |
| `make reinstall` / `make clean` | Recreate / remove `.venv` |
| `make lint` / `make format` | ruff, same rules as CI |

`make install` uses `uv` if it is installed, otherwise `pip` (installing pip into the venv via `ensurepip` when it is missing). If `.venv` was created with a different Python version (e.g. after a Python upgrade), it is detected and recreated automatically.

Scan options can be overridden: `make scan DIR=~/Movies METHOD=combined THRESHOLD=0.9 WORKERS=8 ARGS='--json dups.json'`.

### Terminal UI

`make tui` opens a full-screen UI: paste (or drag & drop) a folder into the directory field, pick a method and threshold, press **Scan** (`ctrl+s`). Results are listed sorted by similarity, the file suggested to keep is highlighted in green.

| Key | Action |
| --- | --- |
| `↑` / `↓` | Select a duplicate pair (full paths shown below the table) |
| `1` / `2` | Play file 1 / file 2 in the default player |
| `r` | Reveal both files in Finder |
| `d` | Move one file of the pair to the **Trash** (asks which; the one not suggested to keep is preselected) |
| `e` | Export results to `video_duplicates_<date>.json` in the scanned folder |
| `l` | Show / hide the scan log |
| `q` | Quit |

The **log panel** at the bottom shows the scan as it happens: every analyzed file (size, duration, fps, whether audio could be read), files that could not be opened and why, each duplicate pair found, pairs that are *similar but just below the threshold* (handy for tuning it), and a summary of how many pairs were compared or skipped. If no audio could be read from any file, it suggests installing `ffmpeg`.

Files are moved to the system Trash (via `send2trash`), never deleted permanently.

> The first start after installation can take up to a minute while OpenCV and librosa are loaded (on macOS the system also scans the freshly installed native libraries). Later starts are fast.

### Command line

```bash
.venv/bin/python video_duplicate_finder.py                       # interactive
.venv/bin/python video_duplicate_finder.py ~/Movies              # defaults, no prompts
.venv/bin/python video_duplicate_finder.py ~/Movies -m combined -t 0.9 -w 8 --json dups.json
.venv/bin/python video_duplicate_finder.py -h
```

### Manual setup (without make)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tui.py
```

### Dependencies
- **opencv-python**: Video processing and frame extraction
- **imagehash**: Perceptual hashing algorithms
- **Pillow**, **numpy**, **scikit-image**: Image processing and analysis
- **librosa**, **soundfile**: Audio fingerprinting
- **tqdm**: CLI progress bars
- **textual**: Terminal UI
- **send2trash**: Safe "move to Trash" from the TUI

### Interactive Configuration

When run without arguments (`make run`), the script will prompt you for:

1. **Directory Path**: Enter the path to scan for video files
2. **Hash Method Selection**:
   - `1` - **pHash** (Perceptual Hash) - *Recommended* - Best for similar content
   - `2` - **dHash** (Difference Hash) - Good for rotated/scaled videos
   - `3` - **Average Hash** - Fastest, basic comparison
   - `4` - **Combined** - All methods together (most accurate)

3. **Similarity Threshold** (0.0-1.0):
   - `0.85` (default) - Balanced accuracy
   - `0.95` - Very strict (fewer false positives)
   - `0.70` - More lenient (may catch more variants)

### Example Output

```
🔍 Found 3 duplicate pairs:
================================================================================

📹 Duplicate Pair #1 (Similarity: 92.5%)
   File 1: /path/to/video1.mp4
           Size: 125.50 MB, Duration: 120.3s
   File 2: /path/to/video1_copy.mp4
           Size: 118.20 MB, Duration: 120.3s
   💡 Suggestion: Keep File 1 (larger size)

💾 Potential space savings: 243.70 MB
```

## Algorithm Details

### How It Works

1. **Frame Extraction**: Samples 15 frames evenly distributed throughout each video
2. **Multi-Method Analysis**:
   - **Perceptual Hashing**: Creates fingerprints resistant to compression changes
   - **Color Histograms**: Analyzes RGB color distribution patterns
   - **Metadata Comparison**: Compares duration, FPS, and file properties

3. **Similarity Scoring**: Combines multiple factors with weights:
   - Hash Similarity (40%) - Primary content comparison
   - Color Histogram (20%) - Visual color patterns
   - Duration Match (15%) - Video length similarity
   - File Size (15%) - Storage size comparison
   - FPS Match (10%) - Frame rate consistency

4. **Smart Filtering**: Only pairs exceeding the similarity threshold are reported

### Supported Formats

- **Video**: .mp4, .avi, .mkv, .mov, .m4v, .wmv, .flv, .webm, .mts, .m2ts
- **Codecs**: Works with any codec supported by OpenCV

## Performance Tips

- **Large Collections**: Use higher similarity thresholds (0.90+) for faster processing
- **Network Drives**: Copy files locally first for better performance
- **Memory Usage**: Processing is optimized for large video collections
- **Accuracy vs Speed**: Combined method is most accurate but slowest

## Troubleshooting

### Common Issues

**`make install` fails with `No module named pip`**
- The existing `.venv` was created without pip (e.g. by `uv`). Install `uv`, or run `make reinstall`.

**`make tui` seems to hang right after start**
- It is most likely still loading OpenCV/librosa - see the note above. If it hangs for several minutes, run `make reinstall` (a `.venv` mixing two Python versions can cause this).

**"No module named cv2"**
```bash
pip install opencv-python
```

**"Cannot read video file"**
- Check if the file is corrupted
- Ensure the codec is supported
- Try converting to a standard format (MP4/H.264)

**"Permission denied"**
- Run with appropriate file permissions
- Check if files are in use by other applications

### Performance Issues
- For large collections (1000+ videos), consider processing in batches
- Use SSD storage for better I/O performance
- Close other video applications during processing

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open source. Please check the repository for license details.
