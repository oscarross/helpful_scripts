#!/usr/bin/env python3

from random import randrange
import argparse
import os
import requests

prefix = "MOCK_"
base_url = "https://picsum.photos"


def make_dir(name):
    try:
        os.makedirs(name)
    except OSError:
        pass
    os.chdir(name)


def download_image(index, width, height):
    random_id = randrange(1, 1000)
    image_url = f'{base_url}/id/{random_id}/{width}/{height}'
    image_data = requests.get(image_url).content
    filename = f'{prefix}{index}.png'

    print(f'⬇️  Downloading - {filename} - {image_url}')

    with open(filename, 'wb') as output:
        output.write(image_data)


def main():
    output_folder = 'output_images'

    parser = argparse.ArgumentParser(description='Image downloader')
    parser.add_argument('--width', '-w', action='store',
                        dest='width', required=False, help='Image width')
    parser.add_argument('--height', '-h', action='store',
                        dest='height', required=False, help='Image height')
    parser.add_argument('--number', '-n', action='store',
                        dest='number', required=True, help='Number of images')

    arguments = parser.parse_args()

    if not arguments.width:
        width = 300
    else:
        width = arguments.width

    if not arguments.height:
        height = 300
    else:
        height = arguments.height

    if not arguments.number:
        number_of_images = 10
    else:
        number_of_images = arguments.number

    print(f'width {width}')
    print(f'height {height}')
    print(f'number_of_images {number_of_images}')

    make_dir(output_folder)

    for index in range(int(number_of_images)):
        download_image(index, width, height)


if __name__ == '__main__':
    main()
