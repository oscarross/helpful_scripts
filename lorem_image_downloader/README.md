# Lorem image downloader

This bash script downloads images from [https://picsum.photos](https://picsum.photos) to created folder `output_images`

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
|**h**|Help|-|

### Examples
With default parameters

`./lorem_image_downloader.sh`

With custom parameters

`./lorem_image_downloader.sh -w 400 -h 500 -n 3 -i 1000`
