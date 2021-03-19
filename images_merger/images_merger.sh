#!/bin/bash

# Required to install
# https://formulae.brew.sh/formula/imagemagick

# Params
BORDER_WIDTH=3
BORDER_COLOR=black
INPUT_FOLDER='./input_images'
OUTPUT_FOLDER='./output_images'
NUMBER_OF_COLUMNS=4

# Constants
GENERATED_FILENAME='merged.png'

# Functions
show_help() {
  cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -c '#323a47' -b 10 -n 2
OPTIONS:
   -w           Border width
   -c           Border color
   -n           Number of columns
   -i           Input folder
   -o           Output folder
   -h           Help
EOF
}

show_install_info() {
  cat <<EOF
❌ Please install "imagemagick"
https://formulae.brew.sh/formula/imagemagick

You can install by brew
"brew install imagemagick"
EOF
}

show_variables() {
  cat <<EOF
============================
Variables:

BORDER_WIDTH="$BORDER_WIDTH"
BORDER_COLOR="$BORDER_COLOR"
NUMBER_OF_COLUMNS="$NUMBER_OF_COLUMNS"
INPUT_FOLDER="$INPUT_FOLDER"
OUTPUT_FOLDER="$OUTPUT_FOLDER"
============================
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

# =============================================

if [[ $(command -v montage) == "" ]]; then
  show_install_info
  exit 1
fi

show_variables

if [ ! -d "$INPUT_FOLDER" ]; then
  echo "❌ Input folder dosen't exists"
  mkdir "$INPUT_FOLDER"
  echo "Input folder created. Please move there images that you want to merge."
  exit 1
fi

echo "🔵 Checking that $OUTPUT_FOLDER exist"
if [ ! -d "$OUTPUT_FOLDER" ]; then
  echo "🔵 Creating folder $OUTPUT_FOLDER"
  mkdir -p "$OUTPUT_FOLDER"
fi

OUTPUT_PATH="./$OUTPUT_FOLDER/$GENERATED_FILENAME"
INPUT_FILES="./$INPUT_FOLDER/*"

echo "🔵 Start merging images"
montage "$INPUT_FILES" -bordercolor "$BORDER_COLOR" -border "$BORDER_WIDTH" -tile "$NUMBER_OF_COLUMNS"x -geometry +0+0 "$OUTPUT_PATH"
montage "$OUTPUT_PATH" -bordercolor "$BORDER_COLOR" -border "$BORDER_WIDTH" -geometry +0+0 "$OUTPUT_PATH"

if [ $? -eq 0 ]; then
  echo "✅ Success: changed pictures are in the folder $OUTPUT_FOLDER"
  exit 0
else
  echo "❌ Failure: there was some problem with $0" >&2
  exit 1
fi
