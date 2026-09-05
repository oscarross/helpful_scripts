#!/bin/bash
set -euo pipefail

# Required to install
# https://formulae.brew.sh/formula/imagemagick
# HEIC decoding needs the libheif delegate (bundled with imagemagick via
# Homebrew; on Debian/Ubuntu also install libheif-examples)

# Params
INPUT_FOLDER='./input_images'
OUTPUT_FOLDER='./output_images'
FORMAT='png'
QUALITY=90

# Functions
show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -f jpg -q 85
OPTIONS:
   -i           Input folder
   -o           Output folder
   -f           Output format: png or jpg
   -q           JPG quality 1-100 (ignored for png)
   -h           Help
EOF
}

show_install_info() {
    cat <<EOF
❌ "convert" can't read HEIC files

Install ImageMagick with HEIC support:

macOS:
  brew install imagemagick libheif

Linux (Debian/Ubuntu):
  sudo apt-get install imagemagick libheif-examples
EOF
}

show_variables() {
    cat <<EOF
============================
Variables:

INPUT_FOLDER="$INPUT_FOLDER"
OUTPUT_FOLDER="$OUTPUT_FOLDER"
FORMAT="$FORMAT"
QUALITY="$QUALITY"
============================
EOF
}

# Get params
while getopts "hi:o:f:q:" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    i) INPUT_FOLDER="$OPTARG" ;;
    o) OUTPUT_FOLDER="$OPTARG" ;;
    f) FORMAT="$OPTARG" ;;
    q) QUALITY="$OPTARG" ;;
    *) shift ;;
    esac
done

# =============================================

if [[ "$FORMAT" != "png" && "$FORMAT" != "jpg" ]]; then
    echo "❌ Format must be 'png' or 'jpg' (got: $FORMAT)" >&2
    exit 1
fi

if [[ $(command -v convert) == "" ]]; then
    echo "❌ Please install \"imagemagick\""
    echo "https://formulae.brew.sh/formula/imagemagick"
    exit 1
fi

if ! convert -list format 2>/dev/null | grep -qi heic; then
    show_install_info
    exit 1
fi

show_variables

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "❌ Input folder doesn't exist"
    mkdir -p "$INPUT_FOLDER"
    echo "Input folder created. Please move there the HEIC images you want to convert."
    exit 1
fi

echo "🔵 Checking that $OUTPUT_FOLDER exist"
if [ ! -d "$OUTPUT_FOLDER" ]; then
    echo "🔵 Creating folder $OUTPUT_FOLDER"
    mkdir -p "$OUTPUT_FOLDER"
fi

shopt -s nocasematch
HEIC_FILES=()
for f in "$INPUT_FOLDER"/*; do
    filename="$(basename "$f")"
    [[ "$filename" == .* ]] && continue
    if [[ "$f" == *.heic || "$f" == *.heif ]]; then
        HEIC_FILES+=("$f")
    fi
done
shopt -u nocasematch

if [[ ${#HEIC_FILES[@]} -eq 0 ]]; then
    echo "❌ No HEIC/HEIF images found in $INPUT_FOLDER"
    exit 1
fi

echo "🔵 Converting ${#HEIC_FILES[@]} image(s) to $FORMAT"
FAILED=0
for f in "${HEIC_FILES[@]}"; do
    FILENAME="$(basename "$f")"
    NEW_FILENAME="${FILENAME%.*}.$FORMAT"
    echo "  $FILENAME -> $NEW_FILENAME"
    if [[ "$FORMAT" == "jpg" ]]; then
        convert "$f" -quality "$QUALITY" "$OUTPUT_FOLDER/$NEW_FILENAME" || FAILED=$((FAILED + 1))
    else
        convert "$f" "$OUTPUT_FOLDER/$NEW_FILENAME" || FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -eq 0 ]]; then
    echo "✅ Success: converted images are in $OUTPUT_FOLDER"
    exit 0
else
    echo "❌ Failure: $FAILED image(s) failed to convert" >&2
    exit 1
fi
