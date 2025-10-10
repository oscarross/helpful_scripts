# Video Duplicate Finder

## Overview

This script helps you find duplicate video files in a specified directory by analyzing their content and identifying similar videos.

## Features

- Calculates a hash for each video file based on its content.
- Compares the hashes to identify duplicate videos.
- Displays information about file duplicates, including file paths and sizes.
- Saves logs to a file for future reference.

## Requirements

Make sure you have Python installed on your system. Additionally, install the required Python packages by running:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

````

## Usage

1. Clone the repository or download the script file.

2. Open a terminal or command prompt.

3. Navigate to the directory containing the script.

4. Run the script using the following command:

```bash
python video_duplicate_finder.py
```

5. Enter the path to the directory containing your video files when prompted.

6. The script will analyze the videos, identify duplicates, and display the results.

## Note

- The script supports common video file formats such as .mp4, .avi, .mkv, and .mov.
- Make sure the `requirements.txt` file is in the same directory as the script.
````
