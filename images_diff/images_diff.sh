#!/usr/bin/env bash

# Required to install
# https://formulae.brew.sh/formula/imagemagick

# Detect OS and set font path
if [[ "$OSTYPE" == "darwin"* ]]; then
    FONT_PATH="/Library/Fonts/OpenSans-Regular.ttf"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    FONT_PATH="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
else
    FONT_PATH=""  # Will use ImageMagick default
fi

# Params
BORDER_WIDTH=3
BORDER_COLOR=black
FIRST_FILE=""
SECOND_FILE=""
ADD_LABELS_OLD_NEW=false
SKIP_GIF=false
SKIP_MERGE=false

OUTPUT_PATH='./output_images'
OUTPUT_FOLDER='GENERATED'
GIF_TEMP_DIR="./gif_temp"

# Cleanup function
cleanup() {
    if [ -d "$GIF_TEMP_DIR" ]; then
        rm -rf "$GIF_TEMP_DIR"
    fi
}

trap cleanup EXIT

# Functions
show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -f before.png -s after.png -l -c '#323a47' -w 10
OPTIONS:
   -f           First file (required)
   -s           Second file (required)
   -l           Add labels OLD / NEW
   -w           Border width (default: 3)
   -c           Border color (default: black)
   -o           Output path (default: ./output_images)
   -g           Skip GIF generation
   -m           Skip merge generation
   -h           Help
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

show_install_info_ffmpeg() {
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

BORDER_WIDTH="$BORDER_WIDTH"
BORDER_COLOR="$BORDER_COLOR"
INPUT_FOLDER="$INPUT_FOLDER"
OUTPUT_PATH="$OUTPUT_PATH"
============================
EOF
}

# Get params
while getopts "hw:c:o:f:s:lgm" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    f) FIRST_FILE="$OPTARG" ;;
    s) SECOND_FILE="$OPTARG" ;;
    w) BORDER_WIDTH="$OPTARG" ;;
    c) BORDER_COLOR="$OPTARG" ;;
    o) OUTPUT_PATH="$OPTARG" ;;
    l) ADD_LABELS_OLD_NEW=true ;;
    g) SKIP_GIF=true ;;
    m) SKIP_MERGE=true ;;
    *) shift ;;
    esac
done

# =============================================

if [[ $(command -v montage) == "" ]]; then
    show_install_info_imagemagick
    exit 1
fi

if [[ $(command -v ffmpeg) == "" ]]; then
    show_install_info_ffmpeg
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

if [ ! -f "$FIRST_FILE" ]; then
    echo "❌ FILE NOT FOUND: $FIRST_FILE"
    exit 1
fi

if [ ! -f "$SECOND_FILE" ]; then
    echo "❌ FILE NOT FOUND: $SECOND_FILE"
    exit 1
fi

echo "🔵 Checking that $OUTPUT_PATH/$OUTPUT_FOLDER exist"
if [ ! -d "$OUTPUT_PATH/$OUTPUT_FOLDER" ]; then
    echo "🔵 Creating folder $OUTPUT_FOLDER"
    mkdir -p "$OUTPUT_PATH/$OUTPUT_FOLDER"
fi

echo "🔵 Delete contents inside $OUTPUT_FOLDER folder"
rm -rf $OUTPUT_PATH/$OUTPUT_FOLDER/* $OUTPUT_PATH/$OUTPUT_FOLDER/.[a-zA-Z0-9]*

INPUT_FILES=($FIRST_FILE $SECOND_FILE)

echo "🔵 Start adding titles to images"

for index in ${!INPUT_FILES[@]}; do
    FILE_PATH="${INPUT_FILES[$index]}"
    FILENAME="$(basename "$FILE_PATH")"
    NEW_FILE_PATH="$OUTPUT_PATH/$OUTPUT_FOLDER/$FILENAME"

    if $ADD_LABELS_OLD_NEW; then
        if [ $index -gt 0 ]; then
            if [ -z "$FONT_PATH" ]; then
                convert "$FILE_PATH" -pointsize 50 -gravity North -background Gold -splice 0x60 -fill black -annotate +0+2 "OLD" "$NEW_FILE_PATH"
            else
                convert "$FILE_PATH" -pointsize 50 -gravity North -background Gold -splice 0x60 -font "$FONT_PATH" -fill black -annotate +0+2 "OLD" "$NEW_FILE_PATH"
            fi
        else
            if [ -z "$FONT_PATH" ]; then
                convert "$FILE_PATH" -pointsize 50 -gravity North -background Gold -splice 0x60 -fill black -annotate +0+2 "NEW" "$NEW_FILE_PATH"
            else
                convert "$FILE_PATH" -pointsize 50 -gravity North -background Gold -splice 0x60 -font "$FONT_PATH" -fill black -annotate +0+2 "NEW" "$NEW_FILE_PATH"
            fi
        fi
    else
        if [ -z "$FONT_PATH" ]; then
            convert "$FILE_PATH" -pointsize 50 -gravity North -background Gold -splice 0x60 -fill black -annotate +0+2 "$FILENAME" "$NEW_FILE_PATH"
        else
            convert "$FILE_PATH" -pointsize 50 -gravity North -background Gold -splice 0x60 -font "$FONT_PATH" -fill black -annotate +0+2 "$FILENAME" "$NEW_FILE_PATH"
        fi
    fi
done

echo "🔵 Start generating diffs"

compare "$FIRST_FILE" "$SECOND_FILE" -quiet -metric AE -fuzz 5% -highlight-color red "$OUTPUT_PATH/$OUTPUT_FOLDER/diff.png" >&- 2>&-

convert '(' "$FIRST_FILE" -flatten -grayscale Rec709Luminance ')' \
    '(' "$SECOND_FILE" -flatten -grayscale Rec709Luminance ')' \
    '(' -clone 0-1 -compose darken -composite ')' \
    -channel RGB -combine "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png"

if $ADD_LABELS_OLD_NEW; then
    if [ -z "$FONT_PATH" ]; then
        convert "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png" -pointsize 50 -gravity North -background Gold -splice 0x60 -fill black -annotate +0+2 "OLD [red] NEW [green].png" "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png"
    else
        convert "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png" -pointsize 50 -gravity North -background Gold -splice 0x60 -font "$FONT_PATH" -fill black -annotate +0+2 "OLD [red] NEW [green].png" "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png"
    fi
else
    if [ -z "$FONT_PATH" ]; then
        convert "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png" -pointsize 50 -gravity North -background Gold -splice 0x60 -fill black -annotate +0+2 "FIRST [red] SECOND [green].png" "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png"
    else
        convert "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png" -pointsize 50 -gravity North -background Gold -splice 0x60 -font "$FONT_PATH" -fill black -annotate +0+2 "FIRST [red] SECOND [green].png" "$OUTPUT_PATH/$OUTPUT_FOLDER/diff_red_green.png"
    fi
fi

if [ -z "$FONT_PATH" ]; then
    convert "$OUTPUT_PATH/$OUTPUT_FOLDER/diff.png" -pointsize 50 -gravity North -background Gold -splice 0x60 -fill black -annotate +0+2 "DIFF" "$OUTPUT_PATH/$OUTPUT_FOLDER/diff.png"
else
    convert "$OUTPUT_PATH/$OUTPUT_FOLDER/diff.png" -pointsize 50 -gravity North -background Gold -splice 0x60 -font "$FONT_PATH" -fill black -annotate +0+2 "DIFF" "$OUTPUT_PATH/$OUTPUT_FOLDER/diff.png"
fi

echo "🔵 Start merging images"
if ! $SKIP_MERGE; then
    ./helpers/merge_images.sh -i "$OUTPUT_PATH/$OUTPUT_FOLDER" -o "$OUTPUT_PATH/$OUTPUT_FOLDER" -n 4 -w "$BORDER_WIDTH" -c "$BORDER_COLOR"
else
    echo "⏭️  Skipping merge (use -m to disable)"
fi

echo "🔵 Start generating gif"
if ! $SKIP_GIF; then
    if [ ! -d "$GIF_TEMP_DIR" ]; then
        mkdir -p "$GIF_TEMP_DIR"
    fi

    COPY_INDEX=0
    for FILE_PATH in "${INPUT_FILES[@]}"; do
        FILENAME="$(basename "$FILE_PATH")"
        cp "$FILE_PATH" "$GIF_TEMP_DIR"
        mv "$GIF_TEMP_DIR/$FILENAME" "$GIF_TEMP_DIR/image$COPY_INDEX.png"
        ((COPY_INDEX++))
    done

    FPS=30
    VIDEO_NAME="temp_video.mp4"
    PALETTE_FILENAME="palette.png"
    ffmpeg -hide_banner -loglevel error -r 3 -i "$GIF_TEMP_DIR/image%01d.png" -c:v libx264 -vf fps=$FPS -pix_fmt yuv420p "$GIF_TEMP_DIR/$VIDEO_NAME"
    ffmpeg -hide_banner -loglevel error -i "$GIF_TEMP_DIR/$VIDEO_NAME" -vf fps="$FPS",scale=-1:800:flags=lanczos,palettegen "$GIF_TEMP_DIR/$PALETTE_FILENAME"
    ffmpeg -hide_banner -loglevel error -i "$GIF_TEMP_DIR/$VIDEO_NAME" -i "$GIF_TEMP_DIR/$PALETTE_FILENAME" -filter_complex "fps=$FPS,scale=-1:800:flags=lanczos[x];[x][1:v]paletteuse" -r "$FPS" "$GIF_TEMP_DIR/diff.gif"
    mv "$GIF_TEMP_DIR/diff.gif" "$OUTPUT_PATH/$OUTPUT_FOLDER/diff.gif"
    echo "✅ GIF created: $OUTPUT_PATH/$OUTPUT_FOLDER/diff.gif"
else
    echo "⏭️  Skipping GIF (use -g to disable)"
fi
