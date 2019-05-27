#!/bin/bash

# Required to install
# https://formulae.brew.sh/formula/ffmpeg

# Helpful sites
# https://medium.com/@colten_jackson/doing-the-gif-thing-on-debian-82b9760a8483

# Params
INPUT_FOLDER='./input_videos'
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

show_install_info() {
cat << EOF
Please install "ffmpeg"
https://formulae.brew.sh/formula/ffmpeg

You can install by brew
"brew install ffmpeg"
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

if [[ $(command -v ffmpeg) == "" ]]; then
    show_install_info
    exit 1
fi

if [ ! -d "$INPUT_FOLDER" ]; then
    echo "❌ Input folder dosen't exists"
    mkdir $INPUT_FOLDER
    echo "Input folder created. Please move there images that you want to merge."
    exit 1
fi

if [ ! -d $OUTPUT_FOLDER ]; then
  mkdir -p $OUTPUT_FOLDER;
fi

generate_palette() {
    ffmpeg -i $1 -vf fps="$FPS",scale=-1:800:flags=lanczos,palettegen $PALETTE_FILENAME
}

generate_gif_from_palette() {
    ffmpeg -i $1 -i $PALETTE_FILENAME -filter_complex "fps=$FPS,scale=-1:800:flags=lanczos[x];[x][1:v]paletteuse" -r "$FPS" $2
}

# Remove whitespaces in filenames
for FILE in $INPUT_FOLDER/*.mp4 $INPUT_FOLDER/*.mov; do 
    mv "$FILE" "${FILE// /_}"; 
done

for FILE in *.mp4 *.mov; do
    echo $FILE
    cd $INPUT_FOLDER
    
    GENERATED_FILENAME=$(basename -- $FILE)
    GENERATED_FILENAME="${GENERATED_FILENAME%.*}.gif"

    generate_palette $FILE
    generate_gif_from_palette $FILE $GENERATED_FILENAME

    if [ $? -ne 0 ]; then
        echo "❌ Can't generate gif"
        exit 2
    fi

    rm $PALETTE_FILENAME

    cd ..
    mv "$INPUT_FOLDER/$GENERATED_FILENAME" "$OUTPUT_FOLDER/$GENERATED_FILENAME"
done

echo "✅ Success gifs are in $OUTPUT_FOLDER"
exit 0