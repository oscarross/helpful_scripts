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

### Examples
With default parameters

`./video_to_gif.sh`

With custom parameters

`./video_to_gif.sh -f 30`

With custom localization 

`./video_to_gif.sh -i "a/input_folder" -o "GIFS"`

### Helpful sites
* [https://medium.com/@colten_jackson/doing-the-gif-thing-on-debian-82b9760a8483](https://medium.com/@colten_jackson/doing-the-gif-thing-on-debian-82b9760a8483)