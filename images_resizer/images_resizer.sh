#!/bin/bash
set -euo pipefail

# Resolve paths relative to this script so it works from any directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Required to install
# https://formulae.brew.sh/formula/imagemagick
# https://github.com/JamieMason/ImageOptim-CLI

# Params
IMAGE_WIDTH=400
IMAGE_HEIGHT=0
INPUT_FOLDER='./input_images/'
OUTPUT_FOLDER='./output_images'
COMPRESSION=false

# Functions
show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -w 400 -a 300 -c
    $0 --tui
OPTIONS:
   --tui        Interactive mode (asks for every option)
   -h           Help
   -w           Images width
   -a           Images height
   -i           Input folder
   -o           Output folder
   -c           Compression
EOF
}

show_install_info_imagemagick() {
    cat <<EOF
❌ Please install "imagemagick"
https://formulae.brew.sh/formula/imagemagick

You can install by brew
"brew install imagemagick"
EOF
}

show_install_info_imageoptim_cli() {
    cat <<EOF
❌ Please install "imageoptim-cli"
https://formulae.brew.sh/formula/imageoptim-cli

You can install by brew
"brew install imageoptim-cli"
EOF
}

show_install_info_imageoptim_app() {
    cat <<EOF
❌ Please install "imageoptim"

You can download app from site:
https://imageoptim.com/
EOF
}

show_variables() {
    cat <<EOF
============================
Variables:

IMAGE_WIDTH="$IMAGE_WIDTH"
IMAGE_HEIGHT="$IMAGE_HEIGHT"
COMPRESSION="$COMPRESSION"
INPUT_FOLDER="$INPUT_FOLDER"
OUTPUT_FOLDER="$OUTPUT_FOLDER"
============================
EOF
}

run_tui() {
    tui_header "Images resizer" "Resizes all images from the input folder (0 keeps the aspect ratio)"
    tui_add_arg -i "$(tui_input "Input folder" "$INPUT_FOLDER")"
    tui_add_arg -o "$(tui_input "Output folder" "$OUTPUT_FOLDER")"
    tui_add_arg -w "$(tui_input_number "Width" "$IMAGE_WIDTH")"
    tui_add_arg -a "$(tui_input_number "Height" "$IMAGE_HEIGHT")"
    if tui_confirm "Compress with ImageOptim?" no; then
        tui_add_arg -c
    fi
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
while getopts "hw:a:i:o:c" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    w) IMAGE_WIDTH="$OPTARG" ;;
    a) IMAGE_HEIGHT="$OPTARG" ;;
    c) COMPRESSION=true ;;
    i) INPUT_FOLDER="$OPTARG" ;;
    o) OUTPUT_FOLDER="$OPTARG" ;;
    *) shift ;;
    esac
done

# =============================================

if [[ $(command -v mogrify) == "" ]]; then
    show_install_info_imagemagick
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

RESIZE=$IMAGE_WIDTH'x'$IMAGE_HEIGHT
if [ $IMAGE_HEIGHT == 0 ]; then
    RESIZE='x'$IMAGE_WIDTH
fi

echo "🔵 Start resizing images"
mogrify -resize "$RESIZE" -path "$OUTPUT_FOLDER" "$INPUT_FOLDER/*"

if $COMPRESSION; then
    if [[ $(command -v imageoptim) == "" ]]; then
        show_install_info_imageoptim_cli
        exit 1
    fi

    echo "🔵 Start compressing images"

    IMAGEOPTIM_APP_EXIST=$(mdfind -name 'kMDItemFSName=="ImageOptim.app"' -onlyin /Volumes/Macintosh\ HD/Applications/ -count)
    if [[ $IMAGEOPTIM_APP_EXIST == 0 ]]; then
        show_install_info_imageoptim_app
        exit 1
    fi
    cd "$OUTPUT_FOLDER"

    if ls *.jpg >/dev/null 2>&1; then
        mogrify -strip -interlace Plane -sampling-factor 4:2:0 -quality 85% *.jpg
    fi

    if ls *.jpeg >/dev/null 2>&1; then
        mogrify -strip -interlace Plane -sampling-factor 4:2:0 -quality 85% *.jpeg
    fi

    if ls *.png >/dev/null 2>&1; then
        mogrify -filter Triangle -define filter:support=2 -unsharp 0.25x0.08+8.3+0.045 -dither None -posterize 136 -quality 82 -define png:compression-filter=5 -define png:compression-level=9 -define png:compression-strategy=1 -define png:exclude-chunk=all -interlace none -colorspace sRGB *.png
    fi

    imageoptim .
fi

if [ $? -eq 0 ]; then
    echo "✅ Success: changed pictures are in the folder $OUTPUT_FOLDER"
    exit 0
else
    echo "❌ Failure: there was some problem with $0" >&2
    exit 1
fi
