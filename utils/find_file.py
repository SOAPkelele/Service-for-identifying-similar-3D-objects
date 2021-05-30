import glob
from pathlib import Path

from bot_app.config import DATASET_PATH


def find_file(file_path: Path):
    file_name = "_".join(file_path.stem.split("_")[1:])
    obj_category = "_".join(file_name.split("_")[:-1])
    root = Path(DATASET_PATH)

    search_path = str(root) + "/" + obj_category + f"/{file_name}*.*"
    files = glob.iglob(search_path, recursive=True)
    for file in files:
        print(file)
        if "preview" in file:
            preview_photo = file
        else:
            mesh_file = file

    return preview_photo, mesh_file
