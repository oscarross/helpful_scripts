# Video to gif

This bash converts all videos (`mp4` or `mov`) from folder `input_videos` to `gifs`.

### Requirement
`ffmpeg`

Documentation
[https://formulae.brew.sh/formula/ffmpeg](https://formulae.brew.sh/formula/ffmpeg)

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**f**|number of FPS|15|
|**i**|Input folder|`./input_videos`|
|**o**|Output folder|`./output_images`|
|**h**|Help|-|

### Interactive mode
Run with `--tui` to be asked for every option instead of passing flags. The script shows the resulting command and asks before running it. Prompts use [gum](https://github.com/charmbracelet/gum) when it is installed (`brew install gum`) and plain terminal prompts otherwise.

`./video_to_gif.sh --tui`

You can also pick this script from the main launcher: `./helpful_scripts.sh` in the repo root.

### Examples
With default parameters

`./video_to_gif.sh`

With custom parameters

`./video_to_gif.sh -f 30`

With custom localization

`./video_to_gif.sh -i "a/input_folder" -o "GIFS"`

### Helpful sites
* [https://medium.com/@colten_jackson/doing-the-gif-thing-on-debian-82b9760a8483](https://medium.com/@colten_jackson/doing-the-gif-thing-on-debian-82b9760a8483)
