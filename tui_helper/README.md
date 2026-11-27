# TUI helper

Shared helper behind the `--tui` (interactive) mode of the bash scripts in this repo and the main launcher `helpful_scripts.sh`. It is not a script to run on its own: scripts `source` it.

It uses [gum](https://github.com/charmbracelet/gum) for nicer prompts when it is installed and falls back to plain `read` prompts otherwise, so nothing extra is required. Works with bash 3.2 (macOS default).

```bash
brew install gum    # optional
TUI_NO_GUM=1 ./images_merger/images_merger.sh --tui   # force plain prompts
```

### Functions
|Function|Description|
|:----|:-----------|
|`tui_header "Title" "Description"`|Prints a header|
|`tui_input "Prompt" "default"`|Asks for text, prints the answer (or the default)|
|`tui_input_required "Prompt" "default"`|Like `tui_input`, asks again while empty|
|`tui_input_number "Prompt" "default"`|Asks again until the answer is a whole number|
|`tui_choose "Prompt" a b c`|Lets the user pick one option (first is the default)|
|`tui_confirm "Question?" [yes\|no]`|Yes/no question, returns 0 for yes|
|`tui_add_arg -f value`|Adds `-f value` to `TUI_ARGS` (skipped when the value is empty); `tui_add_arg -d` adds a bare flag|
|`tui_confirm_run "$0"`|Shows the resulting command and asks whether to run it|

### Adding `--tui` to a script
Define `run_tui()` that asks for options with the functions above and collects them with `tui_add_arg`, then, right before the argument parsing:

```bash
if [[ "${1:-}" == "--tui" ]]; then
    TUI_HELPER="$SCRIPT_DIR/../tui_helper/tui.sh"
    # shellcheck source=../tui_helper/tui.sh
    source "$TUI_HELPER"
    run_tui
    set -- ${TUI_ARGS[@]+"${TUI_ARGS[@]}"}
fi
```

The answers become ordinary flags, so the rest of the script does not change. Remember to add the script to the `SCRIPTS` list in `helpful_scripts.sh`.
