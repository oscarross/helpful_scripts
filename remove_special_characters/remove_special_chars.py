import argparse
import os
import re
import shutil
import sys
import unicodedata


def convert_to_snake_case(name):
    """Konwertuje nazwę pliku do formatu snake_case"""
    # Usuwanie rozszerzenia
    base_name, extension = os.path.splitext(name)

    # Zamiana polskich znaków na standardowe odpowiedniki
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

    # Alternatywna metoda usuwania znaków diakrytycznych dla innych języków
    # Normalizacja i usunięcie znaków diakrytycznych
    base_name = unicodedata.normalize("NFKD", base_name)
    base_name = "".join([c for c in base_name if not unicodedata.combining(c)])

    # Zamiana spacji i innych znaków specjalnych na podkreślniki
    # Usuwanie znaków, które nie są alfanumeryczne ani podkreślnikami
    clean_name = re.sub(r"[^\w\s-]", "", base_name)

    # Zamiana sekwencji białych znaków i myślników na pojedyncze podkreślniki
    clean_name = re.sub(r"[\s-]+", "_", clean_name)

    # Zamiana wielkich liter na małe litery i podkreślniki
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", clean_name)
    snake_case = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    # Usuwanie podwójnych podkreślników
    while "__" in snake_case:
        snake_case = snake_case.replace("__", "_")

    # Usuwanie podkreślników na początku i końcu
    snake_case = snake_case.strip("_")

    return snake_case + extension


def process_directory(input_dir, output_dir):
    """Przetwarza wszystkie pliki w katalogu wejściowym i zapisuje je z nowymi nazwami w katalogu wyjściowym"""
    # Sprawdzenie czy katalogi istnieją
    if not os.path.exists(input_dir):
        print(f"Błąd: Katalog wejściowy '{input_dir}' nie istnieje.")
        sys.exit(1)

    # Tworzenie katalogu wyjściowego, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)

    # Lista do przechowywania informacji o zmianach nazw
    changes = []

    # Przetwarzanie plików
    for root, dirs, files in os.walk(input_dir):
        # Obliczanie relatywnej ścieżki dla zachowania struktury katalogów
        rel_path = os.path.relpath(root, input_dir)
        if rel_path == ".":
            rel_path = ""

        # Tworzenie odpowiedniego podkatalogu w output_dir
        if rel_path:
            target_dir = os.path.join(output_dir, rel_path)
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_dir = output_dir

        # Przetwarzanie każdego pliku
        for file_name in files:
            original_path = os.path.join(root, file_name)
            new_name = convert_to_snake_case(file_name)
            new_path = os.path.join(target_dir, new_name)

            # Kopiowanie pliku z nową nazwą
            shutil.copy2(original_path, new_path)

            # Zapisanie informacji o zmianie
            changes.append((file_name, new_name))

    return changes


def main():
    """Funkcja główna skryptu"""
    parser = argparse.ArgumentParser(
        description="Konwertuje nazwy plików do formatu snake_case i usuwa znaki specjalne"
    )
    parser.add_argument(
        "input_dir", help="Katalog wejściowy z plikami do przetworzenia"
    )
    parser.add_argument(
        "output_dir", help="Katalog wyjściowy dla przetworzonych plików"
    )

    args = parser.parse_args()

    print(f"Przetwarzanie plików z katalogu: {args.input_dir}")
    print(f"Zapisywanie wyników do: {args.output_dir}")

    changes = process_directory(args.input_dir, args.output_dir)

    # Wyświetlanie podsumowania
    print(f"\nPrzetworzono {len(changes)} plików:")
    for old_name, new_name in changes:
        if old_name != new_name:
            print(f"  {old_name} -> {new_name}")
        else:
            print(f"  {old_name} (bez zmian)")


if __name__ == "__main__":
    main()
