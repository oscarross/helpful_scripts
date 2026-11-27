# Images HEIC converter

This bash script converts all HEIC/HEIF images (e.g. photos from an iPhone) from folder `input_images` to PNG or JPG in `output_images`.

### Requirement
* `imagemagick`, built with HEIC/libheif support

### Installation:
* **macOS**

`brew install imagemagick libheif`

* **Linux (Debian/Ubuntu)**

`sudo apt-get install imagemagick libheif-examples`

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**f**|Output format: `png` or `jpg`|png|
|**q**|JPG quality 1-100 (ignored for `png`)|90|
|**i**|Input folder|`./input_images/`|
|**o**|Output folder|`./output_images/`|
|**h**|Help|-|

### Interactive mode
Run with `--tui` to open the terminal UI of this repo on this script: edit the options in a form, see the resulting command and run it with a live log. The flags above keep working as before.

`./images_heic_converter.sh --tui`

You can also pick this script from the full list with `./helpful_scripts_tui.py` in the repo root.

### Examples
With default parameters (HEIC → PNG)

`./images_heic_converter.sh`

Convert to JPG with custom quality

`./images_heic_converter.sh -f jpg -q 85`

With custom folders

`./images_heic_converter.sh -i "iphone_photos" -o "converted"`
