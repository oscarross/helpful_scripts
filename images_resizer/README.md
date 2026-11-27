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
Run with `--tui` to be asked for every option instead of passing flags. The script shows the resulting command and asks before running it. Prompts use [gum](https://github.com/charmbracelet/gum) when it is installed (`brew install gum`) and plain terminal prompts otherwise.

`./images_resizer.sh --tui`

You can also pick this script from the main launcher: `./helpful_scripts.sh` in the repo root.

### Examples
With default parameters

`./images_resizer.sh`

With custom parameters

`./images_resizer.sh -w 400 -a 300 -c`
