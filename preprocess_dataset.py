import os
import subprocess
import csv
from pathlib import Path
import random
import re
import shutil

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and available in PATH"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, 
                                text=True, 
                                check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def extract_label_from_video_path(video_path: Path):
    """Extract label from the parent directory name of the video file."""
    try:
        # The label is the name of the parent directory
        label = video_path.parent.name
        return label
    except Exception as e:
        print(f"Warning: Could not extract label from video path {video_path}: {str(e)}")
        return None

def resize_video(input_path, output_path):
    """Resize video: shorter side to 256px, then center crop to 224x224px."""
    if os.path.exists(output_path):
        print(f"Video already exists, skipping: {output_path}")
        return

    target_short_side = 256
    crop_size = 224

    # ffmpeg filter string to:
    # 1. Scale the shorter side to target_short_side (e.g., 256px), maintaining aspect ratio.
    #    Uses -2 for the other dimension to ensure it's divisible by 2.
    # 2. Center crop to crop_size x crop_size (e.g., 224x224px).
    vf_string = (
        f"scale='if(lt(iw,ih),{target_short_side},-2)':'if(lt(iw,ih),-2,{target_short_side})',"
        f"crop={crop_size}:{crop_size}"
    )

    cmd = [
        'ffmpeg', '-i', str(input_path), # Ensure input_path is a string
        '-vf', vf_string,
        '-c:v', 'libx264', '-preset', 'medium',
        '-c:a', 'copy',  # Copy audio stream without re-encoding
        str(output_path), # Ensure output_path is a string
        '-y'  # Overwrite output file if it exists
    ]
    
    try:
        # Added stdout and stderr capture for better error reporting
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("\\nERROR: ffmpeg not found. Please install ffmpeg and add it to your PATH.")
        print("On Windows: ")
        print("  1. Download from https://ffmpeg.org/download.html or use chocolatey: choco install ffmpeg")
        print("  2. Add the bin folder to your PATH environment variable")
        print("  3. Restart your terminal/command prompt")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\\nERROR processing video: {input_path}")
        print(f"ffmpeg command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"ffmpeg stdout: {e.stdout}")
        print(f"ffmpeg stderr: {e.stderr}")
        exit(1)

def save_original_mapping(videos, output_dir):
    """Save mapping between original and new video paths"""
    mapping_path = Path(output_dir) / 'original.csv'
    with open(mapping_path, 'w') as f:
        for orig_path, new_path, label in videos:
            orig_filename = Path(orig_path).name
            f.write(f"{orig_filename} {new_path} {label}\n")
    print(f"Created original.csv with {len(videos)} video mappings")

def generate_csv_files(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """Generate CSV files with splits using new normalized paths"""
    print("Step 1: Generating CSV files with labels...")
    
    # Get all video files and their labels recursively from all subdirectories
    videos = []
    global_idx = 0
      # Use rglob to recursively search for mp4 files in all subdirectories
    for video_path in Path(input_dir).rglob('*.mp4'):
        label = extract_label_from_video_path(video_path) # MODIFIED: Call updated function
        if label:
            # Generate new normalized filename right from the start
            new_filename = f'video_{global_idx:06d}.mp4'
            rel_path = os.path.join('videos', new_filename)
            videos.append((str(video_path), rel_path, label))
            global_idx += 1
    
    if not videos:
        raise ValueError("No valid video files found with labels in the input directory or its subdirectories")
    
    print(f"Found {len(videos)} valid videos across all subdirectories in {input_dir}")
    
    # Save original path mapping before shuffling
    save_original_mapping(videos, output_dir)
    
    # Shuffle videos
    random.shuffle(videos)
    
    # Split into train/val/test
    n_total = len(videos)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        'train': videos[:n_train],
        'val': videos[n_train:n_train + n_val],
        'test': videos[n_train + n_val:]
    }
    
    # Save CSV files with new paths
    for split_name, split_videos in splits.items():
        csv_path = Path(output_dir) / f'{split_name}.csv'
        with open(csv_path, 'w') as f:
            for _, new_path, label in split_videos:
                f.write(f"{new_path} {label}\n")
        print(f"Created {split_name}.csv with {len(split_videos)} videos")
    
    return splits

def process_videos(splits, output_dir):
    """Process and resize videos based on the splits"""
    print("Step 2: Processing and resizing videos...")
    video_dir = Path(output_dir) / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    
    total_videos = sum(len(videos) for videos in splits.values())
    processed_count = 0
    
    # Process videos for each split
    for split_name, videos in splits.items():
        print(f"\nProcessing {split_name} split ({len(videos)} videos)...")
        
        # Process each video
        for old_path, new_rel_path, label in videos:
            # Get full output path
            new_path = str(Path(output_dir) / new_rel_path)
            
            # Check if input file exists
            if not os.path.exists(old_path):
                print(f"WARNING: Input file does not exist, skipping: {old_path}")
                continue
                
            # Resize video if it doesn't exist
            processed_count += 1
            print(f"[{processed_count}/{total_videos}] Processing {old_path} -> {new_path}")
            resize_video(old_path, new_path)

def update_csv_delimiter(input_dir, output_dir):
    """Update CSV files to use space as delimiter"""
    print("Step 1: Updating CSV files with space delimiter...")
    
    for split_name in ['train', 'val', 'test']:
        csv_path = Path(output_dir) / f'{split_name}.csv'
        if csv_path.exists():
            # Read existing CSV and update delimiter
            videos = []
            with open(csv_path, 'r') as f:
                for line in f:
                    if ',' in line:  # If comma-separated
                        path, label = line.strip().split(',')
                    else:  # If space-separated or other format
                        parts = line.strip().split()
                        path, label = parts[0], parts[-1]
                    videos.append((path, label))
            
            # Write back with space delimiter
            with open(csv_path, 'w') as f:
                for path, label in videos:
                    f.write(f"{path} {label}\n")
            print(f"Updated delimiter in {split_name}.csv")

def preprocess_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Preprocess videos and create train/val/test splits
    Args:
        input_dir: Directory containing original videos
        output_dir: Directory to save processed videos and annotations
        train_ratio: Ratio of videos for training
        val_ratio: Ratio of videos for validation (remaining will be test)
    """
    # Check if ffmpeg is installed
    if not check_ffmpeg_installed():
        print("\nERROR: ffmpeg not found. Please install ffmpeg and add it to your PATH.")
        print("On Windows: ")
        print("  1. Download from https://ffmpeg.org/download.html or use chocolatey: choco install ffmpeg")
        print("  2. Add the bin folder to your PATH environment variable")
        print("  3. Restart your terminal/command prompt")
        exit(1)
        
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting preprocessing of videos from {input_dir} (including all subdirectories)")
    print(f"Output will be saved to {output_dir}")
    
    # Generate splits with normalized paths
    splits = generate_csv_files(input_dir, output_dir, train_ratio, val_ratio)
    
    # Process and resize videos using the normalized paths
    process_videos(splits, output_dir)
    
    print(f"\nPreprocessing complete! Results saved to {output_dir}")
    print(f"- Train videos: {len(splits['train'])}")
    print(f"- Validation videos: {len(splits['val'])}")
    print(f"- Test videos: {len(splits['test'])}")
    print(f"- CSV files and processed videos are ready for VideoMAE training")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory containing original videos')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed videos and annotations')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of videos for training')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of videos for validation')
    args = parser.parse_args()
    
    preprocess_dataset(args.input_dir, args.output_dir, args.train_ratio, args.val_ratio)