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
Run with `--tui` to open the terminal UI of this repo on this script: edit the options in a form, see the resulting command and run it with a live log. The flags above keep working as before.

`./ios_record_simulator.sh --tui`

You can also pick this script from the full list with `./helpful_scripts_tui.py` in the repo root.

### Examples
With default parameters

`./ios_record_simulator.sh`

With custom parameters

`./ios_record_simulator.sh -f 30`

With custom output folder

`./ios_record_simulator.sh -o "a/output_folder/GIFS"`
