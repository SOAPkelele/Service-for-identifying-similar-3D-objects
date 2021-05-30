import glob as glob
import os
from pathlib import Path

import open3d

from utils.image_utils import render_model_preview

if __name__ == "__main__":
    root = "ModelNet40"

    # visualizer which renders image
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window(visible=False, height=224, width=224)

    # tune render options such as background color, lightning and so
    render_option: open3d.visualization.RenderOption = visualizer.get_render_option()
    render_option.light_on = True

    # go through each file in directories
    photo_count = 0
    for type in os.listdir(root):
        for phrase in ["train", "test"]:
            type_path = os.path.join(root, type)
            phrase_path = os.path.join(type_path, phrase)

            files = glob.glob(os.path.join(phrase_path, "*.off"))
            for file in files:
                render_model_preview(Path(file), Path(phrase_path), visualizer=visualizer)
                print(file)
                photo_count += 1

    visualizer.destroy_window()
    print(f"Finished. Created {photo_count} previews")
