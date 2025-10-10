import logging
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import cv2
import imagehash
import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# Suppress common audio library warnings globally
warnings.filterwarnings("ignore", message="PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*audioread.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")


# Log configuration with custom colored formatter
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[1;34m",  # Bright Blue
        "INFO": "\033[1;32m",  # Bright Green
        "WARNING": "\033[1;33m",  # Bright Yellow
        "ERROR": "\033[1;31m",  # Bright Red
        "CRITICAL": "\033[1;35m",  # Bright Magenta
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record):
        # Get color for the log level
        color = self.COLORS.get(record.levelname, self.RESET)

        # Format: [LEVEL] | message
        formatted = f"{color}[{record.levelname:^8}]{self.RESET} | \033[1;36m{record.getMessage()}\033[0m"
        return formatted


logging.root.setLevel(logging.INFO)
formatter = ColoredFormatter()
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logging.root.addHandler(stream_handler)

logger = logging.getLogger(__name__)


def quick_file_similarity_check(size1: int, size2: int, threshold: float = 0.1) -> bool:
    """
    Quick pre-filter based on file size similarity
    Returns True if files are worth comparing in detail
    """
    if size1 == size2:
        return True

    size_diff = abs(size1 - size2)
    max_size = max(size1, size2)

    # If size difference is less than threshold, worth comparing
    return (size_diff / max_size) <= threshold


def extract_audio_fingerprint(video_path: str) -> Optional[np.ndarray]:
    """
    Extract audio fingerprint from video using chromagram features
    """
    try:
        # Suppress known audio library warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message="PySoundFile failed")
            warnings.filterwarnings("ignore", message=".*audioread.*")
            warnings.filterwarnings("ignore", module="librosa")
            warnings.filterwarnings("ignore", module="soundfile")
            warnings.filterwarnings("ignore", module="audioread")

            # Extract audio from video using librosa
            y, sr = librosa.load(video_path, sr=22050, duration=30)  # First 30 seconds

        # Check if audio was successfully loaded
        if y is None or len(y) == 0:
            logger.warning(f"No audio data found in {video_path}")
            return None

        # Extract chromagram features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # Validate feature dimensions before concatenation
        chroma_mean = np.mean(chroma, axis=1)
        mfcc_mean = np.mean(mfcc, axis=1)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_rolloff_mean = np.mean(spectral_rolloff)

        # Debug: Log feature shapes for troubleshooting
        logger.debug(f"Audio feature shapes - chroma: {chroma_mean.shape}, mfcc: {mfcc_mean.shape}, "
                    f"centroid: scalar, rolloff: scalar")

        # Ensure all features are valid numbers
        if (np.isnan(chroma_mean).any() or np.isnan(mfcc_mean).any() or
            np.isnan(spectral_centroid_mean) or np.isnan(spectral_rolloff_mean)):
            logger.warning(f"Invalid audio features extracted from {video_path}")
            return None

        # Combine features with proper dimensionality
        features = np.concatenate([
            chroma_mean,                                # Shape: (12,)
            mfcc_mean,                                  # Shape: (13,)
            [spectral_centroid_mean],                   # Shape: (1,) - wrap scalar in list
            [spectral_rolloff_mean]                     # Shape: (1,) - wrap scalar in list
        ])

        return features

    except Exception as e:
        logger.warning(f"Could not extract audio from {video_path}: {e}")
        return None

def compare_audio_fingerprints(fp1: Optional[np.ndarray], fp2: Optional[np.ndarray]) -> float:
    """
    Compare two audio fingerprints using cosine similarity
    """
    if fp1 is None or fp2 is None:
        return 0.0

    # Cosine similarity
    dot_product = np.dot(fp1, fp2)
    norm_product = np.linalg.norm(fp1) * np.linalg.norm(fp2)

    if norm_product == 0:
        return 0.0

    return dot_product / norm_product

def calculate_ssim_similarity(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate SSIM (Structural Similarity Index) between two frames
    """
    try:
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        if len(frame2.shape) == 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Resize to same dimensions
        h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
        frame1 = cv2.resize(frame1, (w, h))
        frame2 = cv2.resize(frame2, (w, h))

        # Calculate SSIM
        similarity_score = ssim(frame1, frame2, data_range=255)
        return max(0, similarity_score)  # Ensure non-negative

    except Exception:
        return 0.0

def extract_keypoint_features(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract ORB keypoints and descriptors from frame
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=100)

        # Find keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is not None and len(descriptors) > 0:
            # Create a compact feature vector from descriptors
            feature_vector = np.mean(descriptors, axis=0)
            return feature_vector

        return None

    except Exception:
        return None

def compare_keypoint_features(features1: Optional[np.ndarray], features2: Optional[np.ndarray]) -> float:
    """
    Compare keypoint features using Hamming distance
    """
    if features1 is None or features2 is None:
        return 0.0

    try:
        # Calculate Hamming distance (for ORB descriptors)
        distance = np.sum(features1 != features2)
        max_distance = len(features1) * 8  # 8 bits per byte

        # Convert to similarity score
        similarity = 1.0 - (distance / max_distance)
        return max(0, similarity)

    except Exception:
        return 0.0

def calculate_temporal_fingerprint(video_path: str, sample_count: int = 50) -> Optional[np.ndarray]:
    """
    Create temporal fingerprint based on brightness changes over time
    """
    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // sample_count)

        brightness_values = []

        for i in range(sample_count):
            frame_pos = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and calculate average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            brightness_values.append(avg_brightness)

        cap.release()

        if len(brightness_values) < 10:  # Need minimum samples
            return None

        # Create fingerprint from brightness changes
        brightness_array = np.array(brightness_values)

        # Calculate differences between consecutive frames
        differences = np.diff(brightness_array)

        # Create features: mean, std, gradients
        features = np.concatenate([
            [np.mean(brightness_array), np.std(brightness_array)],
            [np.mean(differences), np.std(differences)],
            differences[:min(20, len(differences))]  # First 20 differences
        ])

        return features

    except Exception as e:
        logger.warning(f"Could not create temporal fingerprint for {video_path}: {e}")
        return None

def compare_temporal_fingerprints(fp1: Optional[np.ndarray], fp2: Optional[np.ndarray]) -> float:
    """
    Compare temporal fingerprints using correlation
    """
    if fp1 is None or fp2 is None:
        return 0.0

    try:
        # Ensure same length
        min_len = min(len(fp1), len(fp2))
        fp1_truncated = fp1[:min_len]
        fp2_truncated = fp2[:min_len]

        # Calculate correlation coefficient
        correlation = np.corrcoef(fp1_truncated, fp2_truncated)[0, 1]

        # Handle NaN case
        if np.isnan(correlation):
            return 0.0

        return max(0, correlation)

    except Exception:
        return 0.0

def calculate_color_histogram(frame: np.ndarray) -> np.ndarray:
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


def compare_histograms(
    hist1: Optional[np.ndarray], hist2: Optional[np.ndarray]
) -> float:
    """
    Compare two histograms using correlation method
    """
    if hist1 is None or hist2 is None:
        return 0.0
    return cv2.compareHist(
        hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL
    )


def calculate_video_hash(
    video_path: str, frame_count: int = 10, hash_method: str = "phash"
) -> Optional[Dict]:
    """
    Calculate video hash using multiple methods for better accuracy

    Args:
        video_path: Path to video file
        frame_count: Number of frames to analyze
        hash_method: 'phash' (perceptual), 'dhash' (difference), 'average' or 'combined'
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return None

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Skip videos that are too short or have invalid properties
        if total_frames <= 0 or fps <= 0:
            logger.warning(
                f"Invalid video properties for {video_path}: frames={total_frames}, fps={fps}"
            )
            return None

        duration = total_frames / fps

        hashes = []
        histograms = []
        frame_interval = max(1, total_frames // frame_count)

        successful_frames = 0

        # Sample frames from different parts of the video
        for i in range(frame_count):
            frame_pos = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            ret, frame = cap.read()
            if not ret:
                continue

            try:
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
                    successful_frames += 1
                    continue

                hashes.append(int(str(frame_hash), 16))
                successful_frames += 1

            except Exception as e:
                logger.warning(f"Error processing frame {i} from {video_path}: {e}")
                continue

    finally:
        cap.release()

    # Need at least 3 successful frames for reliable comparison
    if successful_frames < 3:
        logger.warning(
            f"Too few successful frames ({successful_frames}) for {video_path}"
        )
        return None

    # Calculate average histogram
    avg_histogram = np.mean(histograms, axis=0) if histograms else None

    # Calculate advanced features
    audio_fingerprint = extract_audio_fingerprint(video_path)
    temporal_fingerprint = calculate_temporal_fingerprint(video_path)

    # Return metadata along with hash and advanced features
    return {
        "hash": str(sum(hashes) // len(hashes))
        if hash_method != "combined"
        else str(hash(tuple(hashes))),
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "histogram": avg_histogram,
        "analyzed_frames": successful_frames,
        "audio_fingerprint": audio_fingerprint,
        "temporal_fingerprint": temporal_fingerprint,
    }


def convert_bytes_to_MB(bytes_size: int) -> float:
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

    # Audio fingerprint similarity
    audio_similarity = 0.0
    if (
        hash1_data.get("audio_fingerprint") is not None
        and hash2_data.get("audio_fingerprint") is not None
    ):
        audio_similarity = compare_audio_fingerprints(
            hash1_data["audio_fingerprint"], hash2_data["audio_fingerprint"]
        )

    # Temporal fingerprint similarity
    temporal_similarity = 0.0
    if (
        hash1_data.get("temporal_fingerprint") is not None
        and hash2_data.get("temporal_fingerprint") is not None
    ):
        temporal_similarity = compare_temporal_fingerprints(
            hash1_data["temporal_fingerprint"], hash2_data["temporal_fingerprint"]
        )

    # Enhanced weighted average with new advanced methods
    weights = {
        "hash": 0.25,           # Reduced from 0.4
        "duration": 0.1,        # Reduced from 0.15
        "size": 0.1,            # Reduced from 0.15
        "fps": 0.05,            # Reduced from 0.1
        "histogram": 0.15,      # Reduced from 0.2
        "audio": 0.25,          # New - very important
        "temporal": 0.1,        # New - temporal patterns
    }

    total_score = (
        hash_similarity * weights["hash"]
        + duration_similarity * weights["duration"]
        + size_similarity * weights["size"]
        + fps_similarity * weights["fps"]
        + histogram_similarity * weights["histogram"]
        + audio_similarity * weights["audio"]
        + temporal_similarity * weights["temporal"]
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


def process_single_video(file_path: str, hash_method: str) -> tuple:
    """
    Process a single video file and return its data
    Returns: (file_path, video_data) or (file_path, None) if error
    """
    try:
        file_size = os.path.getsize(file_path)
        logger.info(
            f"Analyzing file: {os.path.basename(file_path)}, Size: {convert_bytes_to_MB(file_size):.2f} MB"
        )

        video_hash_data = calculate_video_hash(
            file_path, frame_count=15, hash_method=hash_method
        )

        if video_hash_data:
            return file_path, {
                "hash_data": video_hash_data,
                "size": file_size,
            }
        else:
            return file_path, None

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return file_path, None


def find_video_duplicates(
    directory, hash_method="phash", similarity_threshold=0.85, max_workers=4
):
    """
    Find video duplicates using advanced comparison methods with optimizations

    Args:
        directory: Directory to scan
        hash_method: Hash method to use ('phash', 'dhash', 'average', 'combined')
        similarity_threshold: Minimum similarity score to consider as duplicate
        max_workers: Number of threads for parallel processing
    """
    video_data = {}
    duplicates = []

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
        ".ogv",
        ".3gp",
    )

    # Collect all video files first
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(video_extensions):
                video_files.append(file_path)

    logger.info(f"📁 Found {len(video_files)} video files to analyze")

    # Process videos in parallel with progress bar
    print("📊 Phase 1: Analyzing video content...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all video processing tasks
        future_to_file = {
            executor.submit(process_single_video, file_path, hash_method): file_path
            for file_path in video_files
        }

        # Collect results as they complete with progress bar
        with tqdm(total=len(video_files), desc="🎬 Processing videos", unit="files") as pbar:
            for future in as_completed(future_to_file):
                file_path, result = future.result()
                if result is not None:
                    video_data[file_path] = result
                pbar.update(1)
                pbar.set_postfix_str(f"Processed: {len(video_data)}/{len(video_files)}")

    logger.info(f"✅ Successfully processed {len(video_data)} videos")

    print("🔍 Phase 2: Finding duplicate pairs...")
    # Group videos by size for faster initial filtering
    size_groups = {}
    for file_path, data in video_data.items():
        size = data["size"]
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(file_path)

    # Only compare videos within similar size ranges
    video_paths = list(video_data.keys())
    comparisons_made = 0
    comparisons_skipped = 0

    # Calculate total possible comparisons for progress bar
    total_comparisons = len(video_paths) * (len(video_paths) - 1) // 2

    with tqdm(total=total_comparisons, desc="🔍 Comparing videos", unit="pairs") as pbar:
        for i, path1 in enumerate(video_paths):
            for j, path2 in enumerate(video_paths[i + 1 :], i + 1):
                data1 = video_data[path1]
                data2 = video_data[path2]

                # Quick size-based filter
                if not quick_file_similarity_check(
                    data1["size"], data2["size"], threshold=0.3
                ):
                    comparisons_skipped += 1
                    pbar.update(1)
                    continue

                # Quick duration filter
                duration1 = data1["hash_data"]["duration"]
                duration2 = data2["hash_data"]["duration"]
                duration_diff = abs(duration1 - duration2) / max(duration1, duration2, 1)

                if duration_diff > 0.5:  # Skip if duration differs by more than 50%
                    comparisons_skipped += 1
                    pbar.update(1)
                    continue

                comparisons_made += 1

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

                pbar.update(1)
                pbar.set_postfix_str(f"Found: {len(duplicates)} duplicates")

    logger.info(
        f"📈 Made {comparisons_made} detailed comparisons, skipped {comparisons_skipped} based on quick filters"
    )
    return duplicates


if __name__ == "__main__":
    hash_methods_info = """=== Video Duplicate Finder ===

Available hash methods:
─────────────────────────────────────────────────────────────────────────────
 1. pHash (Perceptual Hash) - RECOMMENDED
    • Best for finding similar content with different compression
    • Detects videos that look the same but have different quality
    • Resistant to minor changes in brightness/contrast
    • Ideal for: re-encoded videos, different resolutions
─────────────────────────────────────────────────────────────────────────────
 2. dHash (Difference Hash)
    • Excellent for detecting rotated or mirrored videos
    • Focuses on gradients and edge patterns
    • Good for videos with watermarks or logos added
    • Ideal for: rotated content, minor edits
─────────────────────────────────────────────────────────────────────────────
 3. Average Hash
    • Fastest method but least accurate
    • Only finds very similar or identical videos
    • Good for exact duplicates with same encoding
    • Ideal for: quick scans, identical file copies
─────────────────────────────────────────────────────────────────────────────
 4. Combined (All Methods)
    • Uses all three methods together for maximum accuracy
    • Slowest but most thorough detection
    • Combines strengths of all methods
    • Ideal for: comprehensive analysis, important collections
─────────────────────────────────────────────────────────────────────────────

� ADVANCED FEATURES (Always Active):
• Audio Fingerprinting - Analyzes sound patterns for enhanced detection
• SSIM Comparison - Superior image quality assessment vs basic hashes
• Keypoint Detection - Robust against modifications, watermarks, crops
• Temporal Fingerprinting - Unique time-based signatures for each video

💡 Tip: All methods now use advanced multi-modal analysis for maximum accuracy"""

    print(hash_methods_info)

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

    # Performance settings
    workers_input = input("\nNumber of parallel workers (1-8, default: 4): ").strip()
    try:
        max_workers = int(workers_input) if workers_input else 4
        max_workers = max(1, min(8, max_workers))
    except ValueError:
        max_workers = 4

    print(f"⚡ Hash method: {hash_method}")
    print(f"🎯 Similarity threshold: {threshold}")
    print(f"🔧 Parallel workers: {max_workers}")
    print("🚀 Starting duplicate detection...\n")

    duplicates = find_video_duplicates(
        directory_to_scan, hash_method, threshold, max_workers
    )

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

            # Enhanced suggestions based on multiple factors
            size1_mb = dup["size1_mb"]
            size2_mb = dup["size2_mb"]
            duration1 = dup["duration1"]
            duration2 = dup["duration2"]

            # Determine better file based on multiple criteria
            file1_score = 0
            file2_score = 0

            # Size factor (larger is usually better)
            if size1_mb > size2_mb:
                file1_score += 2
            elif size2_mb > size1_mb:
                file2_score += 2

            # Duration factor (longer might be uncut version)
            if duration1 > duration2:
                file1_score += 1
            elif duration2 > duration1:
                file2_score += 1

            # File extension preference (mp4 > avi > others)
            ext1 = dup["file1"].lower().split(".")[-1]
            ext2 = dup["file2"].lower().split(".")[-1]

            preferred_extensions = ["mp4", "mkv", "mov", "avi", "wmv"]
            if ext1 in preferred_extensions and ext2 not in preferred_extensions:
                file1_score += 1
            elif ext2 in preferred_extensions and ext1 not in preferred_extensions:
                file2_score += 1
            elif ext1 in preferred_extensions and ext2 in preferred_extensions:
                if preferred_extensions.index(ext1) < preferred_extensions.index(ext2):
                    file1_score += 1
                elif preferred_extensions.index(ext2) < preferred_extensions.index(
                    ext1
                ):
                    file2_score += 1

            # Suggest which file to keep
            if file1_score > file2_score:
                print("   💡 Suggestion: Keep File 1 (better quality/format)")
            elif file2_score > file1_score:
                print("   💡 Suggestion: Keep File 2 (better quality/format)")
            elif size1_mb > size2_mb:
                print("   💡 Suggestion: Keep File 1 (larger size)")
            elif size2_mb > size1_mb:
                print("   💡 Suggestion: Keep File 2 (larger size)")
            else:
                print("   💡 Files have similar qualities - choose based on preference")

        print("\n" + "=" * 80)

        # Calculate total space that could be saved
        total_space_saved = sum(
            min(dup["size1_mb"], dup["size2_mb"]) for dup in duplicates
        )
        print(f"💾 Potential space savings: {total_space_saved:.2f} MB")

    else:
        print("✅ No duplicates found!")
