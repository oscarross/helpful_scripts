#!/bin/bash

# Params
WIDTH=300
HEIGHT=300
FIRST_IMAGE_INDEX=30
NUMBER_OF_IMAGES=10
OUTPUT_FOLDER='./output_images/'
PREFIX="MOCK_"

# Constants
BASE_URL="https://picsum.photos"

# Functions
show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -w 400 -a 500 -n 3
OPTIONS:
   -w           Image width
   -a           Image height
   -n           Number images
   -i           ID number of first image from $BASE_URL
   -o           Output folder
   -p           Prefix name
   -h           Help
EOF
}

show_variables() {
    cat <<EOF
============================
Variables:

WIDTH="$WIDTH"
HEIGHT="$HEIGHT"
NUMBER_OF_IMAGES="$NUMBER_OF_IMAGES"
FIRST_IMAGE_INDEX="$FIRST_IMAGE_INDEX"
PREFIX="$PREFIX"
OUTPUT_FOLDER="$OUTPUT_FOLDER"
============================
EOF
}

# Get params
while getopts "hw:i:a:n:p:o:" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    w) WIDTH="$OPTARG" ;;
    i) FIRST_IMAGE_INDEX="$OPTARG" ;;
    a) HEIGHT="$OPTARG" ;;
    n) NUMBER_OF_IMAGES="$OPTARG" ;;
    o) OUTPUT_FOLDER="$OPTARG" ;;
    p) PREFIX="$OPTARG" ;;
    *) shift ;;
    esac
done

# =============================================

show_variables

echo "🔵 Checking that $OUTPUT_FOLDER exist"
if [ ! -d "$OUTPUT_FOLDER" ]; then
    echo "🔵 Creating folder $OUTPUT_FOLDER"
    mkdir -p "$OUTPUT_FOLDER"
fi

cd "$OUTPUT_FOLDER"

END_NUMBER=$((FIRST_IMAGE_INDEX + NUMBER_OF_IMAGES))

for ((INDEX = "$FIRST_IMAGE_INDEX"; INDEX < "$END_NUMBER"; INDEX++)); do
    IMAGE_LINK="$BASE_URL/id/$INDEX/$WIDTH/$HEIGHT"
    FILENAME=$PREFIX$INDEX.png
    echo "🔵 Downloading image $IMAGE_LINK"
    echo
    curl "$IMAGE_LINK" --output "$FILENAME"
done

echo "🔵 Renaming images"
INDEX=0
for file in ./*; do
    mv "$file" "$PREFIX$INDEX.png"
    INDEX=$((INDEX + 1))
done

echo "✅ Success: downloaded images are in the folder $OUTPUT_FOLDER"
exit 0
