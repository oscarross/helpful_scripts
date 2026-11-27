#!/usr/bin/env python3

import argparse
import os
import sys
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
    image_url = f"{BASE_URL}/id/{random_id}/{width}/{height}"
    image_data = requests.get(image_url, timeout=30).content
    filename = f"{PREFIX}{index}.png"

    print(f"⬇️  Downloading - {filename} - {image_url}")

    with open(filename, "wb") as output:
        output.write(image_data)


def ask(prompt: str, default: str = "") -> str:
    """Ask for a value in the terminal, returning the default on empty input."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled")
        sys.exit(130)
    return answer or default


def ask_int(prompt: str, default: str = "") -> str:
    """Ask until the answer is a positive whole number."""
    while True:
        answer = ask(prompt, default)
        if answer.isdigit() and int(answer) > 0:
            return answer
        print("⚠️  Please enter a whole number greater than 0")


def tui_args() -> list[str]:
    """Interactive mode: ask for every option and return them as CLI flags."""
    print("\n=== Images downloader ===")
    print("Downloads random images from picsum.photos into ./output_images\n")
    args = [
        "-n",
        ask_int("Number of images"),
        "-w",
        ask_int("Image width", "300"),
        "-ht",
        ask_int("Image height", "300"),
    ]
    print(f"\nCommand:\n  {sys.argv[0]} {' '.join(args)}\n")
    if ask("Run it? [Y/n]").lower().startswith("n"):
        print("Cancelled")
        sys.exit(0)
    return args


def main() -> None:
    """Main function to download random images."""
    output_folder = "output_images"

    parser = argparse.ArgumentParser(description="Image downloader")
    parser.add_argument(
        "--width",
        "-w",
        action="store",
        dest="width",
        type=int,
        default=300,
        help="Image width (default: 300)",
    )
    parser.add_argument(
        "--height",
        "-ht",
        action="store",
        dest="height",
        type=int,
        default=300,
        help="Image height (default: 300)",
    )
    parser.add_argument(
        "--number",
        "-n",
        action="store",
        dest="number",
        type=int,
        required=True,
        help="Number of images",
    )

    parser.add_argument(
        "--tui",
        action="store_true",
        help="Interactive mode (asks for every option)",
    )

    argv = sys.argv[1:]
    if argv[:1] == ["--tui"]:
        argv = tui_args()
    args = parser.parse_args(argv)

    print(f"width: {args.width}")
    print(f"height: {args.height}")
    print(f"number_of_images: {args.number}")

    make_dir(output_folder)

    for index in range(args.number):
        download_image(index, args.width, args.height)


if __name__ == "__main__":
    main()
