# Android rotation script

This bash script will rotate you device by adb tools

### Requirement
* `android-platform-tools` 

### Installation:
* **`android-platform-tools`**

`brew cask install android-platform-tools` 

### Parameters
|Name|Description|Default value|
|:----:|:-----------|:-----:|
|**n**|Number of rotations|30|
|**h**|Help|-|

### Examples
With default parameters

`./androird_rotator.sh`

With custom parameters

`./androird_rotator.sh -n 100`
