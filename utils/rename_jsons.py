from pathlib import Path

# Folder containing the JSON files
FOLDER = Path(r"D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\raw\anno")

# Collect JSON files only
json_files = sorted([f for f in FOLDER.iterdir() if f.suffix.lower() == ".json"])

# Safety check
if not json_files:
    raise RuntimeError("No JSON files found.")

# First rename to temporary names to avoid collisions
temp_files = []
for i, f in enumerate(json_files, start=1):
    tmp = f.with_name(f"__tmp__{i:03d}.json")
    f.rename(tmp)
    temp_files.append(tmp)

# Rename to final format: P001.json, P002.json, ...
for i, f in enumerate(temp_files, start=1):
    new_name = f"P{i:03d}_annotations.json"
    f.rename(new_name)

print(f"Renamed {len(temp_files)} files successfully.")
