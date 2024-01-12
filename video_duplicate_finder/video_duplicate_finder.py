import os
import cv2
from PIL import Image
import imagehash
import logging

# Log configuration
log_format = "\033[1;30;41m%(levelname)-8s\033[1;37;40m | \033[1;36;40m%(message)s\033[1;37;40m"  # Colored formatting

logging.root.setLevel(logging.INFO)
formatter = logging.Formatter(log_format)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logging.root.addHandler(stream_handler)

logger = logging.getLogger(__name__)

def calculate_video_hash(video_path, frame_count=5):
    cap = cv2.VideoCapture(video_path)
    hashes = []

    # Capture frames and calculate hash for each frame
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize frame to reduce processing time
        resized_frame = cv2.resize(gray_frame, (64, 64))

        # Calculate hash using imagehash
        frame_hash = imagehash.average_hash(Image.fromarray(resized_frame))
        hashes.append(int(str(frame_hash), 16))  # Convert hash to integer

    # Close video capture
    cap.release()

    # Return the average hash as a string
    return str(sum(hashes) // len(hashes))

def convert_bytes_to_MB(bytes_size):
    return bytes_size / (1024 * 1024)

def find_video_duplicates(directory):
    hash_dict = {}
    duplicates = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                file_size = os.path.getsize(file_path)
                logger.info(f"Analyzing file: {file_path}, Size: {convert_bytes_to_MB(file_size):.2f} MB")
                video_hash = calculate_video_hash(file_path)
                
                if video_hash in hash_dict:
                    duplicate_file_path = hash_dict[video_hash]
                    duplicate_size = os.path.getsize(duplicate_file_path)
                    duplicates.append((file_path, duplicate_file_path, convert_bytes_to_MB(file_size), convert_bytes_to_MB(duplicate_size)))
                else:
                    hash_dict[video_hash] = file_path

    return duplicates

if __name__ == "__main__":
    directory_to_scan = input("Enter the path to the directory with video files: ")
    duplicates = find_video_duplicates(directory_to_scan)

    if duplicates:
        print("Found duplicates:")
        for duplicate_info in duplicates:
            logger.critical(f"Found duplicate: {duplicate_info[0]} and {duplicate_info[1]}")
            print(f"{duplicate_info[0]} ({duplicate_info[2]:.2f} MB)") 
            print(f"{duplicate_info[1]} ({duplicate_info[3]:.2f} MB)")
    else:
        print("No duplicates.")