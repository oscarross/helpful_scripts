# Images downloader

This bash script downloads images from [https://picsum.photos](https://picsum.photos)

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**w**|Image width|300|
|**ht**|Image height|300|
|**n**|Number of images to download|required, no default|

### Interactive mode
Run with `--tui` to be asked for every option instead of passing flags. The script shows the resulting command and asks before running it.

`./image_downloader.py --tui`

You can also pick this script from the main launcher: `./helpful_scripts.sh` in the repo root.

### Examples
With default width/height

`./image_downloader.py -n 5`

With custom parameters

`./image_downloader.py -w 400 -ht 500 -n 3`
