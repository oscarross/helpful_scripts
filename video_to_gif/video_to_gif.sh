#!/bin/bash
set -euo pipefail

# Resolve paths relative to this script (following symlinks, e.g. the copy
# linked from ios_record_simulator/helpers) so it works from any directory
SCRIPT_PATH="$0"
while [ -L "$SCRIPT_PATH" ]; do
    LINK_TARGET="$(readlink "$SCRIPT_PATH")"
    case "$LINK_TARGET" in
    /*) SCRIPT_PATH="$LINK_TARGET" ;;
    *) SCRIPT_PATH="$(dirname "$SCRIPT_PATH")/$LINK_TARGET" ;;
    esac
done
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

# Required to install
# https://formulae.brew.sh/formula/ffmpeg

# Params
INPUT_FOLDER='./input_videos'
OUTPUT_FOLDER='./output_images'
PALETTE_FILENAME="palette.png"
FPS=15

# Functions
show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -f 15
    $0 --tui
OPTIONS:
   --tui        Interactive mode (asks for every option)
   -f           number of FPS
   -i           Input folder
   -o           Output folder
EOF
}

show_install_info() {
    cat <<EOF
Please install "ffmpeg"
https://formulae.brew.sh/formula/ffmpeg

You can install by brew
"brew install ffmpeg"
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

generate_palette() {
    echo "🔵 Generating palette for $1"
    ffmpeg -hide_banner -loglevel error -i "$1" -vf fps="$FPS",scale=-1:800:flags=lanczos,palettegen $INPUT_FOLDER/$PALETTE_FILENAME
}

generate_gif_from_palette() {
    echo "🔵 Generating gif for $1"
    ffmpeg -hide_banner -loglevel error -i "$1" -i $INPUT_FOLDER/$PALETTE_FILENAME -filter_complex "fps=$FPS,scale=-1:800:flags=lanczos[x];[x][1:v]paletteuse" -r "$FPS" "$2"
}

run_tui() {
    tui_header "Video to gif" "Converts every .mov/.mp4 from the input folder to a gif"
    tui_add_arg -i "$(tui_input "Input folder" "$INPUT_FOLDER")"
    tui_add_arg -o "$(tui_input "Output folder" "$OUTPUT_FOLDER")"
    tui_add_arg -f "$(tui_input_number "FPS" "$FPS")"
    tui_confirm_run "$0"
}

# Interactive mode: ask for the options, then parse them like regular flags
if [[ "${1:-}" == "--tui" ]]; then
    TUI_HELPER="$SCRIPT_DIR/../tui_helper/tui.sh"
    if [[ ! -f "$TUI_HELPER" ]]; then
        echo "❌ --tui needs $TUI_HELPER (keep the script inside the repo)" >&2
        exit 1
    fi
    # shellcheck source=../tui_helper/tui.sh
    source "$TUI_HELPER"
    run_tui
    set -- ${TUI_ARGS[@]+"${TUI_ARGS[@]}"}
fi

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

if [[ $(command -v ffmpeg) == "" ]]; then
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

echo "🔵 Removing whitespaces in filenames"
cd "$INPUT_FOLDER"
for f in *; do mv "$f" "$(echo "$f" | tr ' ' '_')"; done
cd ..

while IFS= read -r -d '' FILE; do
    GENERATED_FILENAME=$(basename -- "$FILE")
    GENERATED_FILENAME="${GENERATED_FILENAME%.*}.gif"

    generate_palette "$FILE"
    generate_gif_from_palette "$FILE" "$OUTPUT_FOLDER/$GENERATED_FILENAME"

    if [ $? -ne 0 ]; then
        echo "❌ Can't generate gif"
        exit 2
    fi

    rm "$INPUT_FOLDER/$PALETTE_FILENAME"
done < <(find "$INPUT_FOLDER" -type f \( -name "*.mov" -o -name "*.mp4" \) -print0)

echo "✅ Success gifs are in $OUTPUT_FOLDER"
exit 0
