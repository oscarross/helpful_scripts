#!/usr/bin/env bash

REPO_PATH=''
DRY_RUN=false

# Ignored branches:
# *master
# *main
# *develop
# *release\*
# IGNORED_BRANCHES_REGEX='(^\*|master$|main$|develop$|release\/)'
IGNORED_BRANCHES_REGEX=""
# =============================================

show_help() {
    cat <<EOF
Usage: $0 [options]
EXAMPLE:
    $0 -r './Project/HelloWorldRepo -d'
OPTIONS:
   -r           repo folder
   -d           dry run
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

# Get params
while getopts "hr:d" opt; do
    case "$opt" in
    h)
        show_help
        exit 0
        ;;
    r) REPO_PATH="$OPTARG" ;;
    d) DRY_RUN=true ;;
    *) shift ;;
    esac
done

# =============================================

show_variables

if [ -z "$REPO_PATH" ]; then
    echo "❌ PLEASE SPECIFY REPOSITORY PATH '-r /repository_folder_path'"
    exit 1
fi

cd $REPO_PATH || exit

echo
if [[ $DRY_RUN == true ]]; then
    echo "⚠️  LIST OF BRANCHES THAT WILL BE DELETED:"
else
    echo "⚠️  START DELETING BRANCHES:"
fi
echo

COUNT=0
for branch in $(git branch -r | egrep -v "$IGNORED_BRANCHES_REGEX"); do
    if [[ "$(git log $branch --since "1 hour ago" | wc -l)" -eq 0 ]]; then
        if [[ $DRY_RUN == true ]]; then
            echo "➡️  $branch - will be deleted"
        else
            echo "🔴 DELETING $branch"
            # local_branch_name=$(echo "$branch" | sed 's/origin\///')
            # git branch -d "$local_branch_name"
            # git push origin --delete "$local_branch_name"
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
