from pathlib import Path
import shutil

SOURCE_DIR = Path(r"D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\raw\videos")          # your current structure
TARGET_DIR = Path(r"D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\raw\renamed_videos")  # new flat folder
TARGET_DIR.mkdir(exist_ok=True)

copied = 0
skipped = 0

for sub in sorted(SOURCE_DIR.iterdir()):
    if not sub.is_dir():
        continue

    folder_name = sub.name  # e.g. P001

    # find video file(s) in the folder
    videos = [f for f in sub.iterdir() if f.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi"}]

    if not videos:
        skipped += 1
        continue

    if len(videos) > 1:
        print(f"SKIP (multiple videos): {sub}")
        skipped += 1
        continue

    src = videos[0]
    dst = TARGET_DIR / f"{folder_name}{src.suffix.lower()}"

    if dst.exists():
        print(f"SKIP (already exists): {dst.name}")
        skipped += 1
        continue

    shutil.copy2(src, dst)
    print(f"COPIED: {src} -> {dst}")
    copied += 1

print(f"\nDone. Copied: {copied}, skipped: {skipped}")
