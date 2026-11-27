# Images merger

This bash script merges all images from folder `input_images` to one image.

### Requirement
`imagemagick`

Documentation
[https://formulae.brew.sh/formula/imagemagick](https://formulae.brew.sh/formula/imagemagick)

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**w**|Border width|3|
|**c**|Border color|black|
|**n**|Number of columns|4|
|**i**|Input folder|`./input_images/`|
|**o**|Output folder|`./output_images/`|
|**h**|Help|-|

### Interactive mode
Run with `--tui` to open the terminal UI of this repo on this script: edit the options in a form, see the resulting command and run it with a live log. The flags above keep working as before.

`./images_merger.sh --tui`

You can also pick this script from the full list with `./helpful_scripts_tui.py` in the repo root.

### Examples
With default parameters

`./images_merger.sh`

![default](./examples/merged_default.png)

With custom parameters

`./images_merger.sh -c '#ffffff' -w 20 -n 2 -i "Test"`

![custom](./examples/merged_custom.png)

### Helpful sites
* [https://imagemagick.org/Usage/montage/](https://imagemagick.org/Usage/montage/)
* [https://www.ibm.com/developerworks/library/l-graf2/?ca=dgr-lnxw15GraphicsLine](https://www.ibm.com/developerworks/library/l-graf2/?ca=dgr-lnxw15GraphicsLine)
