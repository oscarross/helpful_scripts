# Merge images

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
|**h**|Help|-|

### Examples
With default parameters

`./merge_images.sh`

![default](./examples/merged_default.png)

With custom parameters

`./merge_images.sh -c '#ffffff' -w 20 -n 2`

![custom](./examples/merged_custom.png)
