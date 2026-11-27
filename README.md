## helpful_scripts
This repo is a compilation of some **helpful scripts** that automate some tasks and make your life easier.

*DISCLAIMER : The Scripts are not tested therefore use it on your own risk !*

## Setup

This repository contains git submodules. After cloning, run:
```bash
git submodule update --init --recursive
```

## Terminal UI

All scripts can be run from one terminal UI (Python 3, stdlib `curses`, no dependencies):
```bash
./helpful_scripts_tui.py
```

Pick a script on the left, edit its options on the right, check the command preview and press `r` to run it. The output streams into the log panel. Interactive scripts (iOS record simulator, Video duplicate finder) get the whole terminal while they run, then the UI comes back.

Every script also accepts `--tui`, which opens the UI straight on that script, e.g. `./images_merger/images_merger.sh --tui`. The existing flags work as before.

Keys: `↑/↓` move, `Tab`/`←→` switch between the list and the options, `Enter` edit an option, `space` toggle or change a choice, `r` run, `x` stop, `d` reset to defaults, `PgUp/PgDn` scroll the log, `q` quit.

File and folder options open a browser: `Enter` opens a folder or picks a file, `←`/`Backspace` goes up, typing filters the list (start with `.` to see hidden files), `~` jumps home, `space` or "Use this folder" picks a folder, `Tab` lets you type the path by hand instead (with `Tab` completion).

Color options (border color) open a color list with a preview of each color; `Tab` there lets you type any ImageMagick color name or a `#hex` value.

Options with typical values (older than, border width, FPS, columns, sizes, JPG quality…) open a list of ready values; `Tab` there lets you type your own.

Scripts run in the directory you start the UI from, so their default `./input_*` and `./output_*` folders end up there.

## Development

This repo lints shell scripts with [ShellCheck](https://www.shellcheck.net/) and Python scripts with [ruff](https://docs.astral.sh/ruff/), run via [pre-commit](https://pre-commit.com/) and checked in CI on every PR. To run them locally before committing:
```bash
pip install pre-commit
pre-commit install
```

### Branch naming

Branches should follow `<type>/<short-description>`, e.g.:
* `feature/` - a new script or new functionality (e.g. `feature/heic-converter`)
* `fix/` - a bug fix (e.g. `fix/images-merger-font`)
* `docs/` - documentation-only changes (e.g. `docs/agent-files`)

## List of scripts in this repository:
### 🏞 Images
* [Images diff](./images_diff) - script to generate diff from two image
* [Images merger](./images_merger) - script to merge all images from input folder
* [Images downloader](./images_downloader) - script to download boiler plate images
* [Images resizer](./images_resizer) - script to resize images
* [Images HEIC converter](./images_heic_converter) - script to convert HEIC/HEIF images to PNG or JPG

### 🎥 Video
* [Video to gif](./video_to_gif) - script to generate gif from video
* [Video duplicate finder](./video_duplicate_finder) - script to find duplicate video files using perceptual hashing, with a terminal UI (`make tui`) and one-command setup (`make install`)

### 📱 iOS
* [iOS record simulator](./ios_record_simulator) - script to automate recording from iOS simulator
* [iOS xcode cleaner](https://github.com/niklasberglund/xcode-clean.sh/tree/8cdc61d6e05859bc1d352fe629a48c32d167ec9e) - **submodule from GITHUB** script to clean xcode
* [iOS unused swift files and variables](https://github.com/PaulTaykalo/swift-scripts) - **submodule from GITHUB** script to find unused swift files and variables

### 💾 GIT
* [GIT delete old branches](./git_delete_old_branches_local_and_remote) - script to delete branches (local and remote) older than passed date

### 📄 Text/Files
* [Remove special characters](./remove_special_characters) - script to convert filenames to snake_case and remove special characters
