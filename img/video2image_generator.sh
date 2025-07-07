#!/bin/bash

# Exit on any error
set -e

# Check for input argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 path/to/video.mp4"
    exit 1
fi

INPUT_FILE="$1"

# Check file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Get base filename without extension or path
BASENAME=$(basename "$INPUT_FILE")
INPUT_FILE_BASENAME="${BASENAME%.*}"

# Create directory structure
mkdir -p "$INPUT_FILE_BASENAME/image_files"
mkdir -p "$INPUT_FILE_BASENAME/video_file"

# Extract images using ffmpeg
ffmpeg -i "$INPUT_FILE" -vf fps=20 "$INPUT_FILE_BASENAME/image_files/%04d.jpg"

# Move original video into video_file/
mv "$INPUT_FILE" "$INPUT_FILE_BASENAME/video_file/"

# Generate image list text file with correct relative paths
find "$INPUT_FILE_BASENAME/image_files" -type f -name '*.jpg' | sort | sed 's|^|../img/|' > "$INPUT_FILE_BASENAME/images.txt"
