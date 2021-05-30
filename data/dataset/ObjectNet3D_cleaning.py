import os
import glob
from pathlib import Path

if __name__ == "__main__":
    root = "ObjectNet3D"
    files_removed = 0
    for path in os.listdir(root):
        type_path = os.path.join(root, path)

        files = glob.glob(os.path.join(type_path, "*.off"))
        files.append(glob.glob(os.path.join(type_path, "*.vty")))
        for file in files:
            file_name = Path(file).name
            file_ext = Path(file).suffix
            if "_" in file_name:
                files_removed += 1
                os.remove(file)
                print(file)
            elif file_ext == ".vty":
                files_removed += 1
                os.remove(file)
                print(file)

    print(f"Removed {files_removed} files")
