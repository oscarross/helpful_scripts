#!/bin/bash

# Required to install
# https://formulae.brew.sh/formula/ffmpeg

# Helpful sites
# https://medium.com/@colten_jackson/doing-the-gif-thing-on-debian-82b9760a8483

# Params
INPUT_FOLDER='./input_images'
OUTPUT_FOLDER='./output_images'
PALETTE_FILENAME="palette.png"
FPS=15

# Menu
show_help() {
cat << EOF
Usage: $0 [options]
EXAMPLE:
    $0 -f 15
OPTIONS:
   -f           number of FPS
EOF
}

while getopts "hf:" opt; 
do
    case "$opt" in
        h) show_help
           exit 0 ;;
        f)  FPS="$OPTARG" ;;
    esac
done

# =============================================

if [ ! -d $OUTPUT_FOLDER ]; then
  mkdir -p $OUTPUT_FOLDER;
fi

generate_palette() {
    ffmpeg -i $1 -vf fps="$FPS",scale=-1:800:flags=lanczos,palettegen $PALETTE_FILENAME
}

generate_gif_from_palette() {
    ffmpeg -i $1 -i $PALETTE_FILENAME -filter_complex "fps=$FPS,scale=-1:800:flags=lanczos[x];[x][1:v]paletteuse" -r "$FPS" $2
}

for FILE in *.mp4 *.mov; do
    cd $INPUT_FOLDER
    
    GENERATED_FILENAME=$(basename -- $FILE)
    GENERATED_FILENAME="${GENERATED_FILENAME%.*}.gif"

    generate_palette $FILE
    generate_gif_from_palette $FILE $GENERATED_FILENAME
    
    rm $PALETTE_FILENAME

    cd ..
    mv "$INPUT_FOLDER/$GENERATED_FILENAME" "$OUTPUT_FOLDER/$GENERATED_FILENAME"
done
