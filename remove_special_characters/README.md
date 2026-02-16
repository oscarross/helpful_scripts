# Remove Special Characters

Script to convert filenames to `snake_case` format and remove special characters (including Polish diacritics).

## Features

- Converts filenames to snake_case
- Removes Polish characters (ą, ć, ę, ł, ń, ó, ś, ź, ż) and replaces them with ASCII equivalents
- Removes diacritical marks from other languages
- Preserves file extensions
- Maintains directory structure

## Usage

```bash
python remove_special_chars.py <input_directory> <output_directory>
```

## Example

```bash
python remove_special_chars.py ./messy_files ./clean_files
```

### Input files:
```
Zdjęcie Z Wakacji.png
Mój Dokument (kopia).pdf
HelloWorldFile.txt
```

### Output files:
```
zdjecie_z_wakacji.png
moj_dokument_kopia.pdf
hello_world_file.txt
```

## Requirements

Python 3.6+

No external dependencies required (uses only standard library).
