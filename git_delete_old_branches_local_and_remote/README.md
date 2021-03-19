# GIT delete old local and remote branches

This bash script merges all images from folder `input_images` to one image.

### Requirement
`imagemagick` 

Documentation
[https://formulae.brew.sh/formula/imagemagick](https://formulae.brew.sh/formula/imagemagick)

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**w**|Border width|3|
|**c**|Border coloe|black|
|**n**|Number of columns|4|
|**i**|Input folder|`./input_images/`|
|**o**|Output folder|`./output_images/`|
|**h**|Help|-|

### Examples
With default parameters

`./merge_images.sh`

![default](./examples/merged_default.png)

With custom parameters

`./merge_images.sh -c '#ffffff' -w 20 -n 2 -i "Test"`

![custom](./examples/merged_custom.png)

### Helpful sites
* [https://imagemagick.org/Usage/montage/](https://imagemagick.org/Usage/montage/)
* [https://www.ibm.com/developerworks/library/l-graf2/?ca=dgr-lnxw15GraphicsLine](https://www.ibm.com/developerworks/library/l-graf2/?ca=dgr-lnxw15GraphicsLine)
