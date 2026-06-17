#!/usr/bin/env bash

# This script merges images using ImageMagick's montage command
# It takes an input folder and creates a merged image with multiple columns

BORDER_WIDTH=3
BORDER_COLOR=black
NUMBER_OF_COLUMNS=2
INPUT_FOLDER=""
OUTPUT_FOLDER=""

show_help() {
  cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -i ./input -o ./output -n 4
OPTIONS:
   -w           Border width
   -c           Border color
   -n           Number of columns
   -i           Input folder
   -o           Output folder
   -h           Help
EOF
}

# Get params
while getopts "hw:c:n:i:o:" opt; do
  case "$opt" in
  h)
    show_help
    exit 0
    ;;
  w) BORDER_WIDTH="$OPTARG" ;;
  c) BORDER_COLOR="$OPTARG" ;;
  n) NUMBER_OF_COLUMNS="$OPTARG" ;;
  i) INPUT_FOLDER="$OPTARG" ;;
  o) OUTPUT_FOLDER="$OPTARG" ;;
  *) shift ;;
  esac
done

if [ -z "$INPUT_FOLDER" ]; then
  echo "❌ Please specify input folder with -i"
  exit 1
fi

if [ -z "$OUTPUT_FOLDER" ]; then
  echo "❌ Please specify output folder with -o"
  exit 1
fi

# Create merged image using montage
montage "$INPUT_FOLDER"/*.png \
  -tile "${NUMBER_OF_COLUMNS}x" \
  -geometry +${BORDER_WIDTH}+${BORDER_WIDTH} \
  -background "$BORDER_COLOR" \
  "$OUTPUT_FOLDER/merged.png" 2>/dev/null

if [ -f "$OUTPUT_FOLDER/merged.png" ]; then
  echo "✅ Merged image created: $OUTPUT_FOLDER/merged.png"
else
  echo "❌ Failed to create merged image"
  exit 1
fi
