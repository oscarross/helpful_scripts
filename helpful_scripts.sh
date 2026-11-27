#!/usr/bin/env bash
set -euo pipefail

# Main launcher: pick a script from this repo and start it in interactive mode.

# Resolve paths relative to this script so it works from any directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Scripts: "Label|path relative to the repo root"
# .sh is run with bash, .py with python3 (both with --tui),
# a directory with "make tui" (video_duplicate_finder has its own TUI)
SCRIPTS=(
    "🏞  Images diff|images_diff/images_diff.sh"
    "🏞  Images merger|images_merger/images_merger.sh"
    "🏞  Images resizer|images_resizer/images_resizer.sh"
    "🏞  Images HEIC converter|images_heic_converter/images_heic_converter.sh"
    "🏞  Images downloader|images_downloader/image_downloader.py"
    "🎥 Video to gif|video_to_gif/video_to_gif.sh"
    "🎥 Video duplicate finder|video_duplicate_finder"
    "📱 iOS record simulator|ios_record_simulator/ios_record_simulator.sh"
    "💾 GIT delete old branches|git_delete_old_branches_local_and_remote/git_delete_old_branches_local_and_remote.sh"
    "📄 Remove special characters|remove_special_characters/remove_special_chars.py"
)

show_help() {
    cat <<EOF
Usage: $0
Shows a menu of the scripts in this repo and starts the chosen one in its
interactive (--tui) mode. Uses gum when installed, plain prompts otherwise.
OPTIONS:
   -h           Help
EOF
}

if [[ "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

# shellcheck source=tui_helper/tui.sh
source "$SCRIPT_DIR/tui_helper/tui.sh"

LABELS=()
for entry in "${SCRIPTS[@]}"; do
    LABELS+=("${entry%%|*}")
done

tui_header "Helpful scripts" "Choose a script to run"
CHOICE="$(tui_choose "Script" "${LABELS[@]}")"

TARGET=""
for entry in "${SCRIPTS[@]}"; do
    if [[ "${entry%%|*}" == "$CHOICE" ]]; then
        TARGET="$SCRIPT_DIR/${entry#*|}"
        break
    fi
done

# The chosen script runs in the directory the launcher was started from,
# so its default ./input and ./output folders end up where you expect them.
case "$TARGET" in
*.sh) exec bash "$TARGET" --tui ;;
*.py) exec python3 "$TARGET" --tui ;;
*) exec make -C "$TARGET" tui ;;
esac
