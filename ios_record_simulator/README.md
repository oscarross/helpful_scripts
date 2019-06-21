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

### Examples
With default parameters

`./ios_record_simulator.sh`

With custom parameters

`./ios_record_simulator.sh -f 30`

With custom localization 

`./ios_record_simulator.sh -o "a/output_folder" -o "GIFS"`