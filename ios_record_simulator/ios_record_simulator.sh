#!/bin/bash

# Params
INPUT_FOLDER='./input_videos'
VIDEONAME="video.mp4"
OUTPUT_FOLDER='./gifs'
FPS=15

# Functions
show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -f 15
OPTIONS:
   -f           number of FPS
   -i           Input folder
   -o           Output folder
EOF
}

show_variables() {
    cat <<EOF
============================
Variables:

FPS="$FPS"
INPUT_FOLDER="$INPUT_FOLDER"
OUTPUT_FOLDER="$OUTPUT_FOLDER"
============================
EOF
}

# Get params

while getopts "hf:i:o:" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    f) FPS="$OPTARG" ;;
    i) INPUT_FOLDER="$OPTARG" ;;
    o) OUTPUT_FOLDER="$OPTARG" ;;
    *) shift ;;
    esac
done

# =============================================

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

LIST_OF_RUNNING_SIMULATORS=$(xcrun simctl list | grep Booted)
NUMBER_OF_RUNNING_SIMULATORS=$(xcrun simctl list | grep Booted | wc -l)

if [ $NUMBER_OF_RUNNING_SIMULATORS == 1 ]; then
    echo "🔵 Start recording on:"
    echo $LIST_OF_RUNNING_SIMULATORS
elif [ $NUMBER_OF_RUNNING_SIMULATORS == 0 ]; then
    echo "❌ You don't have any running simulator"
    exit 1
elif [ $NUMBER_OF_RUNNING_SIMULATORS -gt 2 ]; then
    echo "❌ Too many booted simulators"
    echo $LIST_OF_RUNNING_SIMULATORS
    exit 2
else
    echo "❌ There is some problem with simulator"
    exit 3
fi

echo "Press 'ctrl' + 'c' to stop recording"
xcrun simctl io booted recordVideo --type=mp4 "$VIDEOS_FOLDER/$VIDEONAME"

echo
video_to_gif.sh -i "$VIDEOS_FOLDER" -o "$GIFS_FOLDER" -f "$FPS"

exit 0