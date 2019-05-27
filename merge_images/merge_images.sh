#!/bin/bash

# Required to install
# https://formulae.brew.sh/formula/imagemagick

# Helpful sites
# https://imagemagick.org/Usage/montage/
# https://www.ibm.com/developerworks/library/l-graf2/?ca=dgr-lnxw15GraphicsLine

# Params
BORDER_WIDTH=3
BORDER_COLOR=black
INPUT_FOLDER='./input_images'
OUTPUT_FOLDER='./output_images'
GENERATED_FILENAME='merged.png'
NUMBER_OF_COLUMNS=4

# Menu
show_help() {
cat << EOF
Usage: $0 [options]
EXAMPLE:
    $0 -c '#323a47' -b 10 -n 2
OPTIONS:
   -w           Border width
   -c           Border color
   -n           Number of columns
   -h           Help
EOF
}

show_install_info() {
cat << EOF
Please install "imagemagick"
https://formulae.brew.sh/formula/imagemagick

You can install by brew
"brew install imagemagick"
EOF
}

while getopts "hw:c:n:" opt; 
do
    case "$opt" in
        h) show_help
           exit 0 ;;
        w)  BORDER_WIDTH="$OPTARG" ;;
        c)  BORDER_COLOR="$OPTARG" ;;
        n)  NUMBER_OF_COLUMNS="$OPTARG" ;;
    esac
done

# =============================================

if [[ $(command -v imagemagick) == "" ]]; then
    show_install_info
    exit 1
fi

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "Input folder dosen't exists"
    mkdir $INPUT_FOLDER
    echo "Input folder created. Please move there images that you want to merge."
    exit 1
fi

if [ ! -d $OUTPUT_FOLDER ]; then
  mkdir -p $OUTPUT_FOLDER;
fi

OUTPUT_PATH="./$OUTPUT_FOLDER/$GENERATED_FILENAME"
INPUT_FILES="./$INPUT_FOLDER/*"

montage $INPUT_FILES -bordercolor $BORDER_COLOR -border $BORDER_WIDTH -tile "$NUMBER_OF_COLUMNS"x -geometry +0+0 $OUTPUT_PATH
montage $OUTPUT_PATH -bordercolor $BORDER_COLOR -border $BORDER_WIDTH -geometry +0+0 $OUTPUT_PATH

exit 0