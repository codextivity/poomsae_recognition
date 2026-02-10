import os

folder = r"D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\raw\annotations"  # change this

for filename in os.listdir(folder):
    if filename.endswith(".json") and not filename.endswith("_annotations.json"):
        base = filename[:-5]  # remove ".json"
        new_name = f"{base}_annotations.json"
        print(new_name)

        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
