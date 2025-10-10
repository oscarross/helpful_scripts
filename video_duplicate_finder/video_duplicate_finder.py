import logging
import os

import cv2
import imagehash
import numpy as np
from PIL import Image

# Log configuration
log_format = "\033[1;30;41m%(levelname)-8s\033[1;37;40m | \033[1;36;40m%(message)s\033[1;37;40m"  # Colored formatting

logging.root.setLevel(logging.INFO)
formatter = logging.Formatter(log_format)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logging.root.addHandler(stream_handler)

logger = logging.getLogger(__name__)


def calculate_color_histogram(frame):
    """
    Calculate color histogram for a frame
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate histogram for each channel
    hist_r = cv2.calcHist([rgb_frame], [0], None, [64], [0, 256])
    hist_g = cv2.calcHist([rgb_frame], [1], None, [64], [0, 256])
    hist_b = cv2.calcHist([rgb_frame], [2], None, [64], [0, 256])

    # Normalize histograms
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    return np.concatenate([hist_r, hist_g, hist_b])


def compare_histograms(hist1, hist2):
    """
    Compare two histograms using correlation method
    """
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(
        hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL
    )


def calculate_video_hash(video_path, frame_count=10, hash_method="phash"):
    """
    Calculate video hash using multiple methods for better accuracy

    Args:
        video_path: Path to video file
        frame_count: Number of frames to analyze
        hash_method: 'phash' (perceptual), 'dhash' (difference), 'average' or 'combined'
    """
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    hashes = []
    histograms = []
    frame_interval = max(1, total_frames // frame_count) if total_frames > 0 else 1

    # Sample frames from different parts of the video
    for i in range(frame_count):
        frame_pos = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

        ret, frame = cap.read()
        if not ret:
            break

        # Calculate color histogram
        histogram = calculate_color_histogram(frame)
        histograms.append(histogram)

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame to standard size
        resized_frame = cv2.resize(gray_frame, (64, 64))
        pil_image = Image.fromarray(resized_frame)

        # Calculate hash based on method
        if hash_method == "phash":
            frame_hash = imagehash.phash(pil_image)
        elif hash_method == "dhash":
            frame_hash = imagehash.dhash(pil_image)
        elif hash_method == "average":
            frame_hash = imagehash.average_hash(pil_image)
        elif hash_method == "combined":
            # Combine multiple hash methods for better accuracy
            phash = imagehash.phash(pil_image)
            dhash = imagehash.dhash(pil_image)
            avg_hash = imagehash.average_hash(pil_image)
            # Combine hashes
            combined = str(phash) + str(dhash) + str(avg_hash)
            hashes.append(hash(combined))
            continue

        hashes.append(int(str(frame_hash), 16))

    cap.release()

    if not hashes:
        return None

    # Calculate average histogram
    avg_histogram = np.mean(histograms, axis=0) if histograms else None

    # Return metadata along with hash
    return {
        "hash": str(sum(hashes) // len(hashes))
        if hash_method != "combined"
        else str(hash(tuple(hashes))),
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "histogram": avg_histogram,
    }


def convert_bytes_to_MB(bytes_size):
    return bytes_size / (1024 * 1024)


def calculate_similarity_score(hash1_data, hash2_data, file1_size, file2_size):
    """
    Calculate similarity score between two videos based on multiple factors
    """
    if not hash1_data or not hash2_data:
        return 0

    # Hash similarity (primary factor)
    hash_similarity = 1.0 if hash1_data["hash"] == hash2_data["hash"] else 0.0

    # Duration similarity
    duration_diff = abs(hash1_data["duration"] - hash2_data["duration"])
    duration_similarity = max(
        0, 1 - (duration_diff / max(hash1_data["duration"], hash2_data["duration"], 1))
    )

    # File size similarity
    size_diff = abs(file1_size - file2_size)
    size_similarity = max(0, 1 - (size_diff / max(file1_size, file2_size, 1)))

    # FPS similarity
    fps_diff = abs(hash1_data["fps"] - hash2_data["fps"])
    fps_similarity = max(
        0, 1 - (fps_diff / max(hash1_data["fps"], hash2_data["fps"], 1))
    )

    # Histogram similarity (color distribution)
    histogram_similarity = 0.0
    if (
        hash1_data.get("histogram") is not None
        and hash2_data.get("histogram") is not None
    ):
        histogram_similarity = max(
            0, compare_histograms(hash1_data["histogram"], hash2_data["histogram"])
        )

    # Weighted average with histogram included
    weights = {
        "hash": 0.4,
        "duration": 0.15,
        "size": 0.15,
        "fps": 0.1,
        "histogram": 0.2,
    }
    total_score = (
        hash_similarity * weights["hash"]
        + duration_similarity * weights["duration"]
        + size_similarity * weights["size"]
        + fps_similarity * weights["fps"]
        + histogram_similarity * weights["histogram"]
    )

    return total_score


def are_likely_duplicates(
    hash1_data, hash2_data, file1_size, file2_size, threshold=0.85
):
    """
    Determine if two videos are likely duplicates based on similarity score
    """
    score = calculate_similarity_score(hash1_data, hash2_data, file1_size, file2_size)
    return score >= threshold, score


def find_video_duplicates(directory, hash_method="phash", similarity_threshold=0.85):
    """
    Find video duplicates using advanced comparison methods

    Args:
        directory: Directory to scan
        hash_method: Hash method to use ('phash', 'dhash', 'average', 'combined')
        similarity_threshold: Minimum similarity score to consider as duplicate
    """
    video_data = {}
    duplicates = []
    processed_pairs = set()

    # Supported video extensions
    video_extensions = (
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".m4v",
        ".wmv",
        ".flv",
        ".webm",
        ".mts",
        ".m2ts",
    )

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(video_extensions):
                try:
                    file_size = os.path.getsize(file_path)
                    logger.info(
                        f"Analyzing file: {file_path}, Size: {convert_bytes_to_MB(file_size):.2f} MB"
                    )

                    video_hash_data = calculate_video_hash(
                        file_path, frame_count=15, hash_method=hash_method
                    )

                    if video_hash_data:
                        video_data[file_path] = {
                            "hash_data": video_hash_data,
                            "size": file_size,
                        }
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue

    # Compare all videos with each other
    video_paths = list(video_data.keys())
    for i, path1 in enumerate(video_paths):
        for j, path2 in enumerate(video_paths[i + 1 :], i + 1):
            # Avoid duplicate comparisons
            pair = tuple(sorted([path1, path2]))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            data1 = video_data[path1]
            data2 = video_data[path2]

            is_duplicate, similarity_score = are_likely_duplicates(
                data1["hash_data"],
                data2["hash_data"],
                data1["size"],
                data2["size"],
                similarity_threshold,
            )

            if is_duplicate:
                duplicates.append(
                    {
                        "file1": path1,
                        "file2": path2,
                        "size1_mb": convert_bytes_to_MB(data1["size"]),
                        "size2_mb": convert_bytes_to_MB(data2["size"]),
                        "similarity_score": similarity_score,
                        "duration1": data1["hash_data"]["duration"],
                        "duration2": data2["hash_data"]["duration"],
                    }
                )

    return duplicates


if __name__ == "__main__":
    print("=== Video Duplicate Finder ===")
    print("Available hash methods:")
    print("1. phash (perceptual hash - recommended)")
    print("2. dhash (difference hash)")
    print("3. average (average hash)")
    print("4. combined (all methods combined)")

    directory_to_scan = input(
        "\nEnter the path to the directory with video files: "
    ).strip()

    # Hash method selection
    hash_choice = input("\nSelect hash method (1-4, default: 1): ").strip()
    hash_methods = {"1": "phash", "2": "dhash", "3": "average", "4": "combined"}
    hash_method = hash_methods.get(hash_choice, "phash")

    # Similarity threshold
    threshold_input = input("\nSimilarity threshold (0.0-1.0, default: 0.85): ").strip()
    try:
        threshold = float(threshold_input) if threshold_input else 0.85
        threshold = max(0.0, min(1.0, threshold))
    except ValueError:
        threshold = 0.85

    print(f"\nUsing hash method: {hash_method}")
    print(f"Similarity threshold: {threshold}")
    print("Scanning for duplicates...\n")

    duplicates = find_video_duplicates(directory_to_scan, hash_method, threshold)

    if duplicates:
        print(f"\n🔍 Found {len(duplicates)} duplicate pairs:")
        print("=" * 80)

        for i, dup in enumerate(duplicates, 1):
            print(
                f"\n📹 Duplicate Pair #{i} (Similarity: {dup['similarity_score']:.2%})"
            )
            print(f"   File 1: {dup['file1']}")
            print(
                f"           Size: {dup['size1_mb']:.2f} MB, Duration: {dup['duration1']:.1f}s"
            )
            print(f"   File 2: {dup['file2']}")
            print(
                f"           Size: {dup['size2_mb']:.2f} MB, Duration: {dup['duration2']:.1f}s"
            )

            # Suggest which file to keep
            if dup["size1_mb"] > dup["size2_mb"]:
                print("   💡 Suggestion: Keep File 1 (larger size)")
            elif dup["size2_mb"] > dup["size1_mb"]:
                print("   💡 Suggestion: Keep File 2 (larger size)")
            else:
                print("   💡 Files have similar sizes")

        print("\n" + "=" * 80)

        # Calculate total space that could be saved
        total_space_saved = sum(
            min(dup["size1_mb"], dup["size2_mb"]) for dup in duplicates
        )
        print(f"💾 Potential space savings: {total_space_saved:.2f} MB")

    else:
        print("✅ No duplicates found!")
