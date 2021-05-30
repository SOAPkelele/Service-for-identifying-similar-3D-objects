import json
import os
import zipfile
from pathlib import Path

import numpy as np
import open3d
import torch
from aiogram import types
from aiogram.dispatcher import FSMContext
from loguru import logger
from torch.autograd import Variable

from bot_app.config import MAPPING_EMBEDDINGS_PATH
from data.models.dataset import prepare_one_item
from utils import find_file
from utils.image_utils import render_model_preview, simplify_model, process_model

GET_MODEL = "GET_MODEL_TO_SEARCH"
download_dir = Path("../../data")


async def search_command_handler(message: types.Message, state: FSMContext):
    logger.info(f"User [{message.from_user.id}] wants to compare models")
    await message.answer("Пришлите модель для поиска по базе.")
    await state.set_state(GET_MODEL)


async def get_model_to_search_handler(message: types.Message,
                                      state: FSMContext,
                                      model: torch.nn.Module,
                                      index):
    await state.finish()

    # save document to user_data/{id}/request/*.off
    output_directory = Path(f"data/user_data/{message.from_user.id}/")  # request_{len(os.listdir())}/")
    os.makedirs(output_directory, exist_ok=True)
    num_requests = len(os.listdir(output_directory))
    output_directory = Path(output_directory, f"request_{num_requests + 1}/")
    os.makedirs(output_directory, exist_ok=True)
    mesh_file = Path(output_directory, message.document.file_name)

    # save file
    await message.document.download(destination=str(mesh_file))

    # render preview
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window(visible=False, height=224, width=224)

    # tune render options such as background color, lightning and so
    render_option: open3d.visualization.RenderOption = visualizer.get_render_option()
    render_option.light_on = True
    filename = render_model_preview(mesh_path=Path(mesh_file),
                                    save_directory=Path(output_directory),
                                    visualizer=visualizer)
    visualizer.destroy_window()

    await message.answer_photo(photo=types.InputFile(filename), caption="Вы прислали данный файл, "
                                                                        "идет поиск похожих, подождите.")

    # simplify mesh
    mesh_simplified_path = simplify_model(mesh_path=mesh_file, save_directory=output_directory, add_postfix=True)

    # process model
    processed_file = process_model(mesh_simplified_path, output_directory=output_directory)

    centers, corners, normals, neighbor_index = prepare_one_item(str(processed_file))

    centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
    corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
    normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
    neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))

    _, features = model(centers=centers.unsqueeze(0), corners=corners.unsqueeze(0),
                        normals=normals.unsqueeze(0), neighbor_index=neighbor_index.unsqueeze(0))

    # save embedding
    features = features.detach().cpu()
    output_file = Path(output_directory, f"embedding_{processed_file.stem}.npz")
    np.savetxt(str(output_file), features)

    # get out embedding
    target_feature_vector = np.loadtxt(str(output_file))

    # find nearest neighbors
    nearest_neighbors = index.get_nns_by_vector(target_feature_vector, 5, include_distances=True)

    # load file with mapping of index to file
    with open(MAPPING_EMBEDDINGS_PATH) as mapping_file:
        mapping = json.load(mapping_file)

    # go through neighbors, concat message with file names, their distances and previews
    # add nearest neighbors to zip archive
    # add all this to media_group
    media_group = types.MediaGroup()
    text = ["Найдены следующие модели:"]
    zip_file_path = output_directory / "models.zip"
    zip_models = zipfile.ZipFile(str(zip_file_path), 'w', zipfile.ZIP_DEFLATED)
    for i, a in enumerate(nearest_neighbors[0]):
        print(f"Index {a}, file: {mapping[str(a)]}, distance: {nearest_neighbors[1][i]}")

        file_neighbor = Path(mapping[str(a)])
        distance = round(nearest_neighbors[1][i], 3)
        text.append(f"{'_'.join(file_neighbor.stem.split('_')[1:])}, расстояние: {distance}")

        preview_photo, mesh_file = find_file(Path(mapping[str(a)]))
        zip_models.write(filename=mesh_file, arcname=os.path.basename(mesh_file))
        media_group.attach_photo(types.InputFile(preview_photo))
    zip_models.close()
    await message.answer_media_group(media=media_group)
    await message.answer_document(caption="\n".join(text), document=types.InputFile(zip_file_path))
