#!/usr/bin/env python3

import argparse
import os
from random import randrange

import requests

PREFIX: str = "MOCK_"
BASE_URL: str = "https://picsum.photos"


def make_dir(name: str) -> None:
    """Create directory and change to it."""
    try:
        os.makedirs(name)
    except OSError:
        pass
    os.chdir(name)


def download_image(index: int, width: int, height: int) -> None:
    """Download a random image from picsum.photos."""
    random_id = randrange(1, 1000)
    image_url = f'{BASE_URL}/id/{random_id}/{width}/{height}'
    image_data = requests.get(image_url, timeout=30).content
    filename = f'{PREFIX}{index}.png'

    print(f'⬇️  Downloading - {filename} - {image_url}')

    with open(filename, 'wb') as output:
        output.write(image_data)


def main() -> None:
    """Main function to download random images."""
    output_folder = 'output_images'

    parser = argparse.ArgumentParser(description='Image downloader')
    parser.add_argument('--width', '-w', action='store',
                        dest='width', type=int, default=300, help='Image width (default: 300)')
    parser.add_argument('--height', '-ht', action='store',
                        dest='height', type=int, default=300, help='Image height (default: 300)')
    parser.add_argument('--number', '-n', action='store',
                        dest='number', type=int, required=True, help='Number of images')

    args = parser.parse_args()

    print(f'width: {args.width}')
    print(f'height: {args.height}')
    print(f'number_of_images: {args.number}')

    make_dir(output_folder)

    for index in range(args.number):
        download_image(index, args.width, args.height)


if __name__ == '__main__':
    main()
