import os
import time
from pathlib import Path

from utils.image_utils import simplify_model, process_model

if __name__ == "__main__":

    root = "ModelNet40"
    new_root = "ModelNet40_simplified_1536"
    processed_root = "ModelNet40_processed_1536"

    start = time.time()
    files_processed = 0
    for type in os.listdir(root):
        for phrase in ['train', 'test']:
            type_path = os.path.join(root, type)
            phrase_path = os.path.join(type_path, phrase)

            # check if output directory for simplified files exists
            output_directory = Path(new_root, type, phrase)
            if not os.path.exists(str(output_directory)):
                os.makedirs(output_directory)

            # check if output directory for processed files exists
            output_directory_processed = Path(processed_root, type, phrase)
            if not os.path.exists(str(output_directory_processed)):
                os.makedirs(output_directory_processed)

            files = Path(root, type, phrase).glob("*.off")
            for file in files:
                # simplify mesh
                mesh_simplified_path = simplify_model(mesh_path=file,
                                                      save_directory=output_directory,
                                                      faces=2048)

                process_model(mesh_simplified_path, output_directory=output_directory_processed)
                files_processed += 1
                print(file)

    end = time.time()
    print(f"Finished {files_processed} files.\n"
          f"Took {(end - start) / 60} minutes")
