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
|**h**|Help|-|

### Examples
With default parameters

`./images_resizer.sh`

With custom parameters

`./images_resizer.sh -w 400 -a 300 -c`
