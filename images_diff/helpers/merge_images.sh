#!/usr/bin/env bash
set -euo pipefail

# This script merges images using ImageMagick's montage command
# It takes an input folder and creates a merged image with multiple columns
# Files are processed in the order specified via command line or alphabetically

BORDER_WIDTH=3
BORDER_COLOR=black
NUMBER_OF_COLUMNS=2
INPUT_FOLDER=""
OUTPUT_FOLDER=""
FILE_LIST=()

# Resolve a font for ImageMagick (montage requires one even when no labels are used)
# Kept in sync with _find_font() in images_merger/images_merger.sh
_find_font() {
    for candidate in \
        /System/Library/Fonts/Helvetica.ttc \
        /System/Library/Fonts/Arial.ttf \
        /Library/Fonts/Arial.ttf \
        /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
        /usr/share/fonts/TTF/DejaVuSans.ttf; do
        [[ -f "$candidate" ]] && {
            echo "$candidate"
            return
        }
    done
    # fall back to whatever ImageMagick knows about
    convert -list font 2>/dev/null | awk '/Font:/{print $2; exit}'
}
FONT="$(_find_font)"

show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -i ./input -o ./output -n 4
    $0 -i ./input -o ./output -n 2 -f 'file1.png' 'file2.png'
OPTIONS:
   -w           Border width
   -c           Border color
   -n           Number of columns
   -i           Input folder
   -o           Output folder
   -f           Files to merge (in order, optional - if not specified, all PNG files are used)
   -h           Help
EOF
}

# Get params
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    case "$arg" in
    -h)
        show_help
        exit 0
        ;;
    -w)
        i=$((i + 1))
        BORDER_WIDTH="${!i}"
        ;;
    -c)
        i=$((i + 1))
        BORDER_COLOR="${!i}"
        ;;
    -n)
        i=$((i + 1))
        NUMBER_OF_COLUMNS="${!i}"
        ;;
    -i)
        i=$((i + 1))
        INPUT_FOLDER="${!i}"
        ;;
    -o)
        i=$((i + 1))
        OUTPUT_FOLDER="${!i}"
        ;;
    -f)
        i=$((i + 1))
        # Collect all file arguments until next flag or end
        while [ $i -le $# ]; do
            next_arg="${!i}"
            if [[ "$next_arg" == -* ]]; then
                i=$((i - 1))
                break
            fi
            FILE_LIST+=("$INPUT_FOLDER/$next_arg")
            i=$((i + 1))
        done
        ;;
    esac
    i=$((i + 1))
done

if [ -z "$INPUT_FOLDER" ]; then
    echo "❌ Please specify input folder with -i"
    exit 1
fi

if [ -z "$OUTPUT_FOLDER" ]; then
    echo "❌ Please specify output folder with -o"
    exit 1
fi

# Create merged image using montage
if [ ${#FILE_LIST[@]} -eq 0 ]; then
    # If no files specified, use all PNG files (sorted)
    montage "$INPUT_FOLDER"/*.png \
        -font "$FONT" \
        -tile "${NUMBER_OF_COLUMNS}x" \
        -geometry +${BORDER_WIDTH}+${BORDER_WIDTH} \
        -background "$BORDER_COLOR" \
        "$OUTPUT_FOLDER/merged.png"
else
    # Use specified files in order
    montage "${FILE_LIST[@]}" \
        -font "$FONT" \
        -tile "${NUMBER_OF_COLUMNS}x" \
        -geometry +${BORDER_WIDTH}+${BORDER_WIDTH} \
        -background "$BORDER_COLOR" \
        "$OUTPUT_FOLDER/merged.png"
fi

if [ -f "$OUTPUT_FOLDER/merged.png" ]; then
    echo "✅ Merged image created: $OUTPUT_FOLDER/merged.png"
else
    echo "❌ Failed to create merged image"
    exit 1
fi
