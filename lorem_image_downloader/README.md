# Lorem image downloader

This bash script downloads images from [https://picsum.photos](https://picsum.photos)

### Requirement
`wget` 

Documentation
[https://formulae.brew.sh/formula/wget](https://formulae.brew.sh/formula/wget)

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**w**|Image width|300|
|**a**|Image height|300|
|**n**|Number of images to download|10|
|**i**|ID number of first image to download|30|
|**p**|Prefix name|`MOCK_`|
|**o**|Output folder|`./output_images/`|
|**h**|Help|-|

### Examples
With default parameters

`./lorem_image_downloader.sh`

With custom parameters

`./lorem_image_downloader.sh -w 400 -h 500 -n 3 -i 1000 -o "./Downloads"`
