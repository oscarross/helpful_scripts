# Images downloader

This bash script downloads images from [https://picsum.photos](https://picsum.photos)

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**w**|Image width|300|
|**ht**|Image height|300|
|**n**|Number of images to download|required, no default|

### Examples
With default width/height

`./image_downloader.py -n 5`

With custom parameters

`./image_downloader.py -w 400 -ht 500 -n 3`
