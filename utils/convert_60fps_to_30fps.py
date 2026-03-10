"""
Downsample 60 FPS Videos to 30 FPS

This script identifies 60 FPS videos and converts them to 30 FPS
by keeping every other frame (frame skipping).
"""

import cv2
from pathlib import Path
from tqdm import tqdm
import shutil


def check_video_fps(video_path):
    """Check the FPS of a video"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def downsample_video_to_30fps(input_path, output_path):
    """
    Downsample a 60 FPS video to 30 FPS by keeping every other frame

    Args:
        input_path: Path to input video (60 FPS)
        output_path: Path to save output video (30 FPS)
    """
    cap = cv2.VideoCapture(str(input_path))

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing: {input_path.name}")
    print(f"  Original FPS: {fps:.2f}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")

    # Create video writer for 30 FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        30.0,  # Target FPS
        (width, height)
    )

    frame_count = 0
    saved_count = 0

    # Process frames
    with tqdm(total=total_frames, desc="Downsampling") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Keep every other frame for 60→30 FPS conversion
            if frame_count % 2 == 0:
                out.write(frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()

    print(f"  ✓ Downsampled: {saved_count} frames saved")
    print(f"  Output: {output_path.name}")

    return saved_count


def process_all_videos(video_dir, output_dir=None, backup_originals=True):
    """
    Find and downsample all 60 FPS videos

    Args:
        video_dir: Directory containing videos
        output_dir: Where to save processed videos (None = replace originals)
        backup_originals: Whether to backup original 60 FPS videos
    """
    video_dir = Path(video_dir)

    # Find all videos
    video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    video_files = [f for f in video_dir.iterdir()
                   if f.suffix in video_extensions]

    if not video_files:
        print(f"No videos found in {video_dir}")
        return

    print(f"\n{'=' * 60}")
    print(f"FPS ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Found {len(video_files)} videos")

    # Analyze FPS
    fps_summary = {}
    videos_60fps = []

    for video_file in video_files:
        fps = check_video_fps(video_file)
        fps_rounded = round(fps)

        if fps_rounded not in fps_summary:
            fps_summary[fps_rounded] = []
        fps_summary[fps_rounded].append(video_file.name)

        # Identify 60 FPS videos
        if fps_rounded >= 59:  # 59-61 FPS counts as 60 FPS
            videos_60fps.append(video_file)

    # Print summary
    print(f"\nFPS Distribution:")
    for fps, files in sorted(fps_summary.items()):
        print(f"  {fps} FPS: {len(files)} videos")
        if fps <30:
            for file in files[:10]:  # Show first 3
                print(f"    - {file}")
        if fps >= 59:
            for file in files[:10]:  # Show first 3
                print(f"    - {file}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 3} more")

    if not videos_60fps:
        print(f"\n✓ All videos are 30 FPS or compatible. No conversion needed!")
        return

    print(f"\n{'=' * 60}")
    print(f"CONVERSION PLAN")
    print(f"{'=' * 60}")
    print(f"Videos to convert: {len(videos_60fps)}")

    # Setup directories
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        replace_mode = False
    else:
        # Will replace originals
        replace_mode = True
        if backup_originals:
            backup_dir = video_dir / '60fps_originals_backup'
            backup_dir.mkdir(exist_ok=True)
            print(f"\nBackup directory: {backup_dir}")

    # Confirm
    print(f"\nMode: {'Replace originals' if replace_mode else 'Create new files'}")
    if replace_mode and backup_originals:
        print(f"Originals will be backed up to: 60fps_originals_backup/")

    response = input("\nProceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return

    # Process videos
    print(f"\n{'=' * 60}")
    print(f"CONVERTING VIDEOS")
    print(f"{'=' * 60}")

    for video_file in videos_60fps:
        if replace_mode:
            # Create temp output
            temp_output = video_dir / f"temp_30fps_{video_file.name}"

            # Downsample
            downsample_video_to_30fps(video_file, temp_output)

            # Backup original
            if backup_originals:
                backup_path = backup_dir / video_file.name
                shutil.copy2(video_file, backup_path)
                print(f"  ✓ Backed up to: {backup_path.name}")

            # Replace original
            video_file.unlink()
            temp_output.rename(video_file)
            print(f"  ✓ Replaced: {video_file.name}")

        else:
            # Save to output directory
            output_path = output_dir / video_file.name
            downsample_video_to_30fps(video_file, output_path)

    print(f"\n{'=' * 60}")
    print(f"CONVERSION COMPLETE")
    print(f"{'=' * 60}")
    print(f"✓ Converted {len(videos_60fps)} videos to 30 FPS")

    if replace_mode and backup_originals:
        print(f"✓ Originals backed up to: {backup_dir}")

    print(f"\nNext steps:")
    print(f"  1. Verify videos play correctly")
    print(f"  2. Re-run keypoint extraction: python preprocessing/extract_keypoints.py")
    print(f"  3. Continue with your pipeline")
    print()


def main():
    """Main function"""
    import sys
    sys.path.append(str(Path(__file__).parent))

    from configs.paths import Paths

    print("\n" + "=" * 60)
    print("VIDEO FPS CONVERTER (60 FPS → 30 FPS)")
    print("=" * 60)

    # Process videos in raw videos directory
    process_all_videos(
        video_dir=Paths.RAW_VIDEOS,
        output_dir=r"D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\raw\drive-download",  # None = replace originals
        backup_originals=True  # Keep backups
    )


if __name__ == "__main__":
    main()