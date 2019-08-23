#!/bin/bash

# Params
VIDEONAME="video.mp4"
OUTPUT_FOLDER='./gifs'
VIDEOS_FOLDER='./videos'
FPS=15

# Functions
show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -f 15
OPTIONS:
   -f           number of FPS
   -o           Output folder
EOF
}

show_variables() {
    cat <<EOF
============================
Variables:

FPS="$FPS"
OUTPUT_FOLDER="$OUTPUT_FOLDER"
============================
EOF
}

# Get params

while getopts "hf:o:" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    f) FPS="$OPTARG" ;;
    o) OUTPUT_FOLDER="$OPTARG" ;;
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

echo "🔵 Checking that $VIDEOS_FOLDER exist"
if [ ! -d "$VIDEOS_FOLDER" ]; then
    echo "🔵 Creating folder $VIDEOS_FOLDER"
    mkdir -p "$VIDEOS_FOLDER"
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
./helpers/video_to_gif.sh -i "$VIDEOS_FOLDER" -o "$OUTPUT_FOLDER" -f "$FPS"

exit 0