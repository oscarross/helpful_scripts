#!/usr/bin/env bash
#
# Shared helper for the "--tui" interactive mode of the scripts in this repo.
# It is sourced (not executed) by a script, which then calls the tui_* functions
# below to ask for its options and collects them into TUI_ARGS as regular CLI
# flags. The script then runs exactly as if those flags were typed by hand.
#
# Uses gum (https://github.com/charmbracelet/gum) when it is installed and falls
# back to plain bash prompts otherwise. Set TUI_NO_GUM=1 to force the fallback.
# Works with bash 3.2 (macOS default).

TUI_ARGS=()
TUI_USE_GUM=false

# Stops the whole script, also when called from inside $(...).
# $$ is the PID of the main script even in a subshell.
tui_abort() {
    echo "Cancelled" >&2
    kill -TERM $$ 2>/dev/null
    exit 130
}

if [[ "${TUI_NO_GUM:-0}" != "1" ]] && command -v gum >/dev/null 2>&1; then
    TUI_USE_GUM=true
fi

if [[ ! -t 0 ]] && [[ ! -r /dev/tty ]]; then
    echo "❌ --tui needs an interactive terminal" >&2
    exit 1
fi

# tui_header "Title" "One line description"
tui_header() {
    if $TUI_USE_GUM; then
        gum style --border rounded --padding "0 2" --border-foreground 212 --bold "$1" "${2:-}" >&2
    else
        printf '\n=== %s ===\n' "$1" >&2
        [[ -n "${2:-}" ]] && printf '%s\n' "$2" >&2
        printf '\n' >&2
    fi
    return 0
}

# value=$(tui_input "Prompt" "default")
# Prints the answer, or the default when the answer is empty.
tui_input() {
    local prompt="$1" default="${2:-}" answer=""
    if $TUI_USE_GUM; then
        answer=$(gum input --header "$prompt" --placeholder "$default" --value "$default") || tui_abort
    else
        if [[ -n "$default" ]]; then
            read -r -e -p "$prompt [$default]: " answer </dev/tty || tui_abort
        else
            read -r -e -p "$prompt: " answer </dev/tty || tui_abort
        fi
    fi
    printf '%s\n' "${answer:-$default}"
}

# value=$(tui_input_required "Prompt" "default")
# Like tui_input, but asks again until the answer is not empty.
tui_input_required() {
    local value=""
    while [[ -z "$value" ]]; do
        value=$(tui_input "$@")
        [[ -z "$value" ]] && echo "⚠️  This value is required" >&2
    done
    printf '%s\n' "$value"
}

# value=$(tui_input_number "Prompt" "default")
# Asks again until the answer is a non-negative integer.
tui_input_number() {
    local value=""
    while true; do
        value=$(tui_input "$@")
        [[ "$value" =~ ^[0-9]+$ ]] && break
        echo "⚠️  Please enter a whole number" >&2
    done
    printf '%s\n' "$value"
}

# value=$(tui_choose "Prompt" option1 option2 ...)
# The first option is the default.
tui_choose() {
    local prompt="$1" answer="" i option
    shift
    if $TUI_USE_GUM; then
        gum choose --header "$prompt" "$@" || tui_abort
        return 0
    fi
    printf '%s\n' "$prompt" >&2
    i=1
    for option in "$@"; do
        printf '  %d) %s\n' "$i" "$option" >&2
        i=$((i + 1))
    done
    while true; do
        read -r -p "Choose 1-$# [1]: " answer </dev/tty || tui_abort
        answer="${answer:-1}"
        if [[ "$answer" =~ ^[0-9]+$ ]] && [ "$answer" -ge 1 ] && [ "$answer" -le $# ]; then
            break
        fi
        echo "⚠️  Please enter a number from 1 to $#" >&2
    done
    shift $((answer - 1))
    printf '%s\n' "$1"
}

# if tui_confirm "Question?" [yes|no]; then ...
# The second argument is the default answer (yes when left out).
tui_confirm() {
    local prompt="$1" default="${2:-yes}" answer=""
    if $TUI_USE_GUM; then
        if [[ "$default" == "yes" ]]; then
            gum confirm "$prompt" && return 0
        else
            gum confirm --default=false "$prompt" && return 0
        fi
        # 1 means "no", anything else (130 = Ctrl+C) cancels
        [[ $? -eq 1 ]] && return 1
        tui_abort
    fi
    if [[ "$default" == "yes" ]]; then
        read -r -p "$prompt [Y/n]: " answer </dev/tty || tui_abort
        [[ ! "$answer" =~ ^[Nn] ]]
    else
        read -r -p "$prompt [y/N]: " answer </dev/tty || tui_abort
        [[ "$answer" =~ ^[Yy] ]]
    fi
}

# tui_add_arg -f value   -> adds "-f value" to TUI_ARGS (skipped when value is empty)
# tui_add_arg -d         -> adds the flag "-d" on its own
tui_add_arg() {
    if [[ $# -eq 1 ]]; then
        TUI_ARGS+=("$1")
    elif [[ -n "$2" ]]; then
        TUI_ARGS+=("$1" "$2")
    fi
    return 0
}

# tui_confirm_run "./script.sh"
# Shows the command that the answers add up to and asks whether to run it.
tui_confirm_run() {
    local cmd="$1" arg
    for arg in ${TUI_ARGS[@]+"${TUI_ARGS[@]}"}; do
        cmd="$cmd $(printf '%q' "$arg")"
    done
    printf '\nCommand:\n  %s\n\n' "$cmd" >&2
    if ! tui_confirm "Run it?"; then
        echo "Cancelled" >&2
        exit 0
    fi
}
