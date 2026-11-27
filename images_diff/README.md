# Diff images

Script to generate visual diffs from two images with support for labels, custom colors, and multi-format output (PNG, GIF, merged montage).

### Features
- ✅ Cross-platform (macOS, Linux)
- ✅ Add labels to images with customizable titles
- ✅ Custom border width and color
- ✅ Generate diff visualizations (red/green composite)
- ✅ Create merged montage of all images
- ✅ Generate animated GIF for quick comparison
- ✅ Input validation and error handling

### Parameters

|Name|Description|Default|Required|
|:----:|:-----------|:-----:|:-----:|
|**f1**|First image file|-|Yes|
|**f2**|Second image file|-|Yes|
|**w**|Border width|3|No|
|**c**|Border color|black|No|
|**o**|Output path|`./output_images`|No|
|**--crop-top**|Pixels to remove from the top of both images before comparison|0|No|
|**l**|Add labels to images|disabled|No|
|**t1**|Custom title for first image|OLD (when -l used)|No|
|**t2**|Custom title for second image|NEW (when -l used)|No|
|**g**|Skip GIF generation|disabled|No|
|**m**|Skip merge generation|disabled|No|
|**h**|Show help|-|No|

### Examples

**Basic usage**
```bash
./images_diff.sh -f1 first.png -f2 second.png
```

**With labels and custom colors**
```bash
./images_diff.sh -f1 first.png -f2 second.png -l -c '#323a47' -w 10
```

**With custom title names**
```bash
./images_diff.sh -f1 first.png -f2 second.png -l -t1 'BEFORE' -t2 'AFTER'
```

**Quick test (skip GIF & merge)**
```bash
./images_diff.sh -f1 first.png -f2 second.png -l -g -m
```

**Ignore the top 100 pixels (for example, the status bar)**
```bash
./images_diff.sh -f1 first.png -f2 second.png -l --crop-top 100
```

Cropping applies to both images before labels, diffs, montage, and GIF generation.
Original files are not modified. The value must be a non-negative integer smaller than the height of each image.
Omit the option or use `--crop-top 0` to keep the full images.

**Show help**
```bash
./images_diff.sh -h
```

### Example result

**Input images**

|Name|Image|
|:----:|:-----------|
|`first.png`|![after](./examples/first.png)|
|`second.png`|![after](./examples/second.png)|

**Output images**

|Name|Description|
|:----:|:-----------|
|`diff_red_green.png`|Red/green composite diff visualization|
|`diff.png`|AE metric diff with red highlights|
|`merged.png`|Montage of all 5 images (before, after, diff, diff_red_green, with labels)|
|`diff.gif`|Animated GIF alternating between images|

### Output Structure

Results are saved in `./output_images/GENERATED/` by default (or `<path>/GENERATED/` with `-o <path>`).
Different images are a successful comparison, not an error. Real comparison errors stop the script and are printed to the terminal.
For GIF encoding, odd-sized images are padded by at most one pixel per dimension without resizing their content.

```
output_images/GENERATED/
├── before.png              # Original first image with title
├── after.png               # Original second image with title
├── diff.png                # AE metric diff
├── diff_red_green.png      # Red/green visualization
├── merged.png              # Montage of all 5 images
└── diff.gif                # Animated comparison
```

### Installation

Requires ImageMagick and FFmpeg:
```bash
# macOS
brew install imagemagick ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install imagemagick ffmpeg

# Linux (Fedora)
sudo dnf install ImageMagick ffmpeg
```
