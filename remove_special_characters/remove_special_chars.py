import argparse
import os
import re
import shutil
import sys
import unicodedata


def convert_to_snake_case(name: str) -> str:
    """Convert filename to snake_case format"""
    # Extract extension
    base_name, extension = os.path.splitext(name)

    # Replace Polish characters with ASCII equivalents
    polish_chars = {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    }

    for polish_char, replacement in polish_chars.items():
        base_name = base_name.replace(polish_char, replacement)

    # Alternative method for removing diacritical marks from other languages
    # Normalize and remove combining characters
    base_name = unicodedata.normalize("NFKD", base_name)
    base_name = "".join([c for c in base_name if not unicodedata.combining(c)])

    # Replace spaces and special characters with underscores
    # Remove characters that are not alphanumeric or underscores
    clean_name = re.sub(r"[^\w\s-]", "", base_name)

    # Replace sequences of whitespace and hyphens with single underscores
    clean_name = re.sub(r"[\s-]+", "_", clean_name)

    # Convert to lowercase with underscores between words
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", clean_name)
    snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    # Remove double underscores
    while "__" in snake_case:
        snake_case = snake_case.replace("__", "_")

    # Remove leading and trailing underscores
    snake_case = snake_case.strip("_")

    return snake_case + extension


def process_directory(input_dir: str, output_dir: str) -> list[tuple[str, str]]:
    """Process all files in input directory and save them with new names in output directory"""
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # List to store rename information
    changes: list[tuple[str, str]] = []

    # Process files
    for root, dirs, files in os.walk(input_dir):
        # Calculate relative path to preserve directory structure
        rel_path = os.path.relpath(root, input_dir)
        if rel_path == ".":
            rel_path = ""

        # Create corresponding subdirectory in output_dir
        if rel_path:
            target_dir = os.path.join(output_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_dir = output_dir

        # Process each file
        for file_name in files:
            original_path = os.path.join(root, file_name)
            new_name = convert_to_snake_case(file_name)
            new_path = os.path.join(target_dir, new_name)

            # Copy file with new name
            shutil.copy2(original_path, new_path)

            # Store rename information
            changes.append((file_name, new_name))

    return changes


def main() -> None:
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Convert filenames to snake_case format and remove special characters"
    )
    parser.add_argument(
        "input_dir", help="Input directory with files to process"
    )
    parser.add_argument(
        "output_dir", help="Output directory for processed files"
    )

    args = parser.parse_args()

    print(f"Processing files from: {args.input_dir}")
    print(f"Saving results to: {args.output_dir}")

    changes = process_directory(args.input_dir, args.output_dir)

    # Display summary
    print(f"\nProcessed {len(changes)} files:")
    for old_name, new_name in changes:
        if old_name != new_name:
            print(f"  {old_name} -> {new_name}")
        else:
            print(f"  {old_name} (unchanged)")


if __name__ == "__main__":
    main()
