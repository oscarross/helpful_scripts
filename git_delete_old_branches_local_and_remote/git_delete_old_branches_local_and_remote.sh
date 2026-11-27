#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script so it works from any directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

REPO_PATH=''
DRY_RUN=false
SINCE="6 months ago"

# Ignored branches:
# *master
# *main
# *develop
# *release\*
IGNORED_BRANCHES_REGEX='(^\*|master$|main$|develop$|release\/)'
# =============================================

show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -r './Project/HelloWorldRepo -s "4 months ago" -d'
    $0 --tui
OPTIONS:
   --tui        Interactive mode (asks for every option)
   -r           repo folder
   -d           dry run
   -s           older than '(6 months | 2 weeks | 3 days | 2 hours  | 30 minutes | 59 seconds) ago'
   -h           Help
EOF
}

show_variables() {
    cat <<EOF
============================
Variables:

REPO_PATH="$REPO_PATH"
DRY_RUN=$DRY_RUN
============================
EOF
}

run_tui() {
    tui_header "GIT delete old branches" "Deletes local and remote branches with no commits since a given date"
    tui_add_arg -r "$(tui_input_required "Repository folder" "${REPO_PATH:-.}")"
    tui_add_arg -s "$(tui_input "Older than (e.g. 6 months ago, 2 weeks ago)" "$SINCE")"
    if tui_confirm "Dry run (only list the branches)?"; then
        tui_add_arg -d
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
while getopts "hr:ds:" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    r) REPO_PATH="$OPTARG" ;;
    d) DRY_RUN=true ;;
    s) SINCE="$OPTARG" ;;
    *) shift ;;
    esac
done

# =============================================

show_variables

if [ -z "$REPO_PATH" ]; then
    echo "❌ PLEASE SPECIFY REPOSITORY PATH '-r /repository_folder_path'"
    exit 1
fi

cd "$REPO_PATH" || exit

echo
if [[ $DRY_RUN == true ]]; then
    echo "⚠️  LIST OF BRANCHES THAT WILL BE DELETED:"
else
    echo "⚠️  START DELETING BRANCHES:"
fi
echo

COUNT=0
for branch in $(git branch -r | grep -E -v "$IGNORED_BRANCHES_REGEX"); do
    if [[ "$(git log "$branch" --since "$SINCE" | wc -l)" -eq 0 ]]; then
        if [[ $DRY_RUN == true ]]; then
            echo "➡️  $branch - will be deleted"
        else
            echo "🔴 DELETING $branch"
            local_branch=$(echo "$branch" | sed 's/origin\///')
            git branch -d "$local_branch"
            git push origin --delete "$local_branch"
        fi
        ((COUNT = COUNT + 1))
    fi
done

echo '============================'
echo NUMBER OF DELETED BRANCHES: "$COUNT"

echo '============================'
echo "BRANCHES"
echo '============================'

for branch in $(git branch -r); do
    echo "💎 $branch"
done
