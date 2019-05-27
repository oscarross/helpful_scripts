#!/bin/bash

# Required to install
# https://formulae.brew.sh/formula/wget

# Params
WIDTH=300
HEIGHT=300
FIRST_IMAGE_INDEX=30
NUMBER_OF_IMAGES=10
BASE_URL="https://picsum.photos"

OUTPUT_FOLDER='./output_images/'
PREFIX="MOCK_"

# Menu

show_help() {
cat << EOF
Usage: $0 [options]
EXAMPLE:
    $0 -w 400 -h 500 -n 3
OPTIONS:
   -w           Image width
   -a           Image height
   -n           Number images
   -i           ID number of first image from $BASE_URL
   -h           Help
EOF
}

show_install_info() {
cat << EOF
Please install "wget"
https://formulae.brew.sh/formula/wget

You can install by brew
"brew install wget"
EOF
}

while getopts "hw:i:a:n:" opt; 
do
    case "$opt" in
        h) show_help
           exit 0 ;;
        w)  WIDTH="$OPTARG" ;;
        i)  FIRST_IMAGE_INDEX="$OPTARG" ;;
        a)  HEIGHT="$OPTARG" ;;
        n)  NUMBER_OF_IMAGES="$OPTARG" ;;
    esac
done

# =============================================

if [[ $(command -v wget) == "" ]]; then
    show_install_info
    exit 1
fi

if [ ! -d $OUTPUT_FOLDER ]; then
  mkdir -p $OUTPUT_FOLDER;
fi

cd $OUTPUT_FOLDER

LINK=$BASE_URL/$WIDTH/$HEIGHT

END_NUMBER=$((FIRST_IMAGE_INDEX+NUMBER_OF_IMAGES))

for (( INDEX=$FIRST_IMAGE_INDEX; INDEX<=$END_NUMBER; INDEX++ ))
do
	IMAGE_LINK=$LINK?image=$INDEX
	FILENAME=$PREFIX$INDEX.png
	wget $IMAGE_LINK -O $FILENAME
done

INDEX=0
for file in ./* 
do
    mv "$file" "$PREFIX$INDEX.png"
    INDEX=$((INDEX + 1))
done

exit 0