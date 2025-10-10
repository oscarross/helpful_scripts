# Video Duplicate Finder

## Overview

An advanced Python tool that finds duplicate video files in a specified directory using multiple sophisticated comparison methods including perceptual hashing, color histogram analysis, and metadata comparison for highly accurate duplicate detection.

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

## Requirements

Make sure you have Python 3.7+ installed on your system. Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies
- **opencv-python**: Video processing and frame extraction
- **imagehash**: Perceptual hashing algorithms
- **Pillow**: Image processing
- **numpy**: Numerical operations for histogram analysis

## Usage

### Quick Start

1. Clone the repository or download the script files:
```bash
git clone <repository-url>
cd video_duplicate_finder
```

2. Set up the environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the script:
```bash
python video_duplicate_finder.py
```

### Interactive Configuration

The script will prompt you for:

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
