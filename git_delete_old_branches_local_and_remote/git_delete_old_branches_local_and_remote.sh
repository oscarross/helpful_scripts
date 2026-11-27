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
   --tui        Open the terminal UI on this script
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

# Interactive mode: open the terminal UI of this repo on this script
if [[ "${1:-}" == "--tui" ]]; then
    exec python3 "$SCRIPT_DIR/../helpful_scripts_tui.py" --script git_delete_old_branches_local_and_remote
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
