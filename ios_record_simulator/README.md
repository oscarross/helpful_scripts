# iOS record simulator

This script record video from booted simulator and after that script will convert this video to `gif`.

### Requirement
script `video_to_gif.sh`

[location of script](./../video_to_gif)

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**f**|number of FPS|15|
|**o**|Output folder|`./gifs`|
|**h**|Help|-|

### Interactive mode
Run with `--tui` to be asked for every option instead of passing flags. The script shows the resulting command and asks before running it. Prompts use [gum](https://github.com/charmbracelet/gum) when it is installed (`brew install gum`) and plain terminal prompts otherwise.

`./ios_record_simulator.sh --tui`

You can also pick this script from the main launcher: `./helpful_scripts.sh` in the repo root.

### Examples
With default parameters

`./ios_record_simulator.sh`

With custom parameters

`./ios_record_simulator.sh -f 30`

With custom output folder

`./ios_record_simulator.sh -o "a/output_folder/GIFS"`
