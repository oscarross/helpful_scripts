#!/bin/bash

# Required to install
# https://formulae.brew.sh/formula/imagemagick

# Params
BORDER_WIDTH=3
BORDER_COLOR=black
FIRST_FILE=""
SECOND_FILE=""

OUTPUT_FOLDER='./output_images'

# Constants
GENERATED_FILENAME='merged.png'

# Functions
show_help() {
  cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -c '#323a47' -b 10
OPTIONS:
   -f           First file
   -s           Second file
   -w           Border width
   -c           Border color
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
INPUT_FOLDER="$INPUT_FOLDER"
OUTPUT_FOLDER="$OUTPUT_FOLDER"
============================
EOF
}

# Get params
while getopts "hw:c:o:f:s:" opt; do
  case "$opt" in
  h)
    show_help
    exit 0
    ;;
  f) FIRST_FILE="$OPTARG" ;;
  s) SECOND_FILE="$OPTARG" ;;
  w) BORDER_WIDTH="$OPTARG" ;;
  c) BORDER_COLOR="$OPTARG" ;;
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

if [ -z "$FIRST_FILE" ]; then
  echo "❌ PLEASE SPECIFY FIRST FILE BY ADDING '-f filepath'"
  exit 1
fi

if [ -z "$SECOND_FILE" ]; then
  echo "❌ PLEASE SPECIFY SECOND FILE BY ADDING '-s filepath'"
  exit 1
fi

echo "🔵 Checking that $OUTPUT_FOLDER exist"
if [ ! -d "$OUTPUT_FOLDER" ]; then
  echo "🔵 Creating folder $OUTPUT_FOLDER"
  mkdir -p "$OUTPUT_FOLDER"
fi

INPUT_FILES=($FIRST_FILE $SECOND_FILE)

echo "🔵 Start adding titles to images"

for FILE_PATH in ${INPUT_FILES[*]} 
do
  FILENAME="$(basename $FILE_PATH)"
  NEW_FILE_PATH="$OUTPUT_FOLDER/$FILENAME"
  convert $FILE_PATH -pointsize 40 -gravity North -background Gold -splice 0x50 -annotate +0+2 "$FILENAME" "$NEW_FILE_PATH"
done

echo "🔵 Start generating diffs"

compare $FIRST_FILE $SECOND_FILE -quiet -metric AE -fuzz 5% -highlight-color red "$OUTPUT_FOLDER/diff.png" >&- 2>&-

convert '(' $FIRST_FILE -flatten -grayscale Rec709Luminance ')' \
        '(' $SECOND_FILE -flatten -grayscale Rec709Luminance ')' \
        '(' -clone 0-1 -compose darken -composite ')' \
        -channel RGB -combine "$OUTPUT_FOLDER/diff_red_green.png"

convert "$OUTPUT_FOLDER/diff_red_green.png" -pointsize 40 -gravity North -background Gold -splice 0x50 -annotate +0+2 "diff_red_green.png" "$OUTPUT_FOLDER/diff_red_green.png"
convert "$OUTPUT_FOLDER/diff.png" -pointsize 40 -gravity North -background Gold -splice 0x50 -annotate +0+2 "diff.png" "$OUTPUT_FOLDER/diff.png"

echo "🔵 Start merging images"
./helpers/merge_images.sh -i $OUTPUT_FOLDER -o $OUTPUT_FOLDER -n 4 >&- 2>&-

echo "🔵 Start generating gif"
convert -delay 50 $FIRST_FILE $SECOND_FILE -loop 0 "$OUTPUT_FOLDER/diff.gif"