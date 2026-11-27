# Images resizer

This bash script resizes all images from folder `input_images` to `output_images` folder.

### Requirement
* `imagemagick`
* `ImageOptim-CLI`
* `ImageOptim.app`

### Installation:
* **`imagemagick`**

`brew install imagemagick`

* **`ImageOptim-CLI`**

`brew install imageoptim-cli`

* **`ImageOptim.app`**

Download app from site [https://imageoptim.com/](https://imageoptim.com/)
and move it to `Applications` folder

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**w**|Image width|400|
|**a**|Image height|-|
|**c**|Compression|false|
|**i**|Input folder|`./input_images/`|
|**o**|Output folder|`./output_images/`|
|**h**|Help|-|

### Interactive mode
Run with `--tui` to open the terminal UI of this repo on this script: edit the options in a form, see the resulting command and run it with a live log. The flags above keep working as before.

`./images_resizer.sh --tui`

You can also pick this script from the full list with `./helpful_scripts_tui.py` in the repo root.

### Examples
With default parameters

`./images_resizer.sh`

With custom parameters

`./images_resizer.sh -w 400 -a 300 -c`
