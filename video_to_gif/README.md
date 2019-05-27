# Video to gif

This bash converts all videos (`mp4` or `mov`) from folder `input_videos` to `gifs`.

### Requirement
`ffmpeg` 

Documentation
[https://formulae.brew.sh/formula/ffmpeg](https://formulae.brew.sh/formula/ffmpeg)

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**f**|FPS|15|
|**h**|Help|-|

### Examples
With default parameters

`./video_to_gif.sh`

With custom parameters

`./video_to_gif.sh -f 30`
