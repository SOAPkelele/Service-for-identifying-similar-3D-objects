import os
from typing import List
from pathlib import Path

import numpy as np
import open3d
import torch
from aiogram import types
from loguru import logger
from aiogram.dispatcher import FSMContext
from torch.autograd import Variable
from scipy import spatial
from data.models.dataset import prepare_one_item
from utils import render_model_preview, simplify_model, process_model

GET_MODELS = "GET_MODELS_TO_COMPARE"
download_dir = Path("../../data")


async def compare_command_handler(message: types.Message, state: FSMContext):
    logger.info(f"User [{message.from_user.id}] wants to compare models")
    await message.answer("Пришлите 2 модели для сравнения.")
    await state.set_state(GET_MODELS)


async def get_models_to_compare_handler(message: types.Message,
                                        model_files: List[types.Message],
                                        state: FSMContext,
                                        model: torch.nn.Module):
    await state.finish()

    # save document to user_data/{id}/request/*.off
    output_directory = Path(f"data/user_data/{message.from_user.id}/")  # request_{len(os.listdir())}/")
    os.makedirs(output_directory, exist_ok=True)
    num_requests = len(os.listdir(output_directory))
    output_directory = Path(output_directory, f"request_{num_requests + 1}/")
    os.makedirs(output_directory, exist_ok=True)

    # first mesh file
    # mesh_file_1 = Path(output_directory, "1_" + message.document.file_name)
    # mesh_file_2 = Path(output_directory, "2_" + message.document.file_name)

    # save files
    for i, file in enumerate(model_files, start=1):
        print(file)
        print(i)
        if i == 1:
            mesh_file_1 = Path(output_directory, "1_" + file.document.file_name)
            await file.document.download(destination=str(mesh_file_1))
        if i == 2:
            mesh_file_2 = Path(output_directory, "2_" + file.document.file_name)
            await file.document.download(destination=str(mesh_file_2))

    # render previews
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window(visible=False, height=224, width=224)

    # tune render options such as background color, lightning and so
    render_option: open3d.visualization.RenderOption = visualizer.get_render_option()
    render_option.light_on = True
    preview_1 = render_model_preview(mesh_path=Path(mesh_file_1),
                                     save_directory=Path(output_directory),
                                     visualizer=visualizer)
    preview_2 = render_model_preview(mesh_path=Path(mesh_file_2),
                                     save_directory=Path(output_directory),
                                     visualizer=visualizer)
    visualizer.destroy_window()

    # send previews of files
    render_photos = types.MediaGroup()
    render_photos.attach_photo(types.InputFile(preview_1),
                               caption="Вы прислали данные файлы, определяем их похожесть...")
    render_photos.attach_photo(types.InputFile(preview_2))
    await message.answer_media_group(media=render_photos)

    # simplify meshes
    mesh_simplified_path_1 = simplify_model(mesh_path=mesh_file_1, save_directory=output_directory, add_postfix=True)
    mesh_simplified_path_2 = simplify_model(mesh_path=mesh_file_2, save_directory=output_directory, add_postfix=True)

    # process first model
    processed_file_1 = process_model(mesh_simplified_path_1, output_directory=output_directory)
    centers, corners, normals, neighbor_index = prepare_one_item(str(processed_file_1))

    centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
    corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
    normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
    neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))

    _, features = model(centers=centers.unsqueeze(0), corners=corners.unsqueeze(0),
                        normals=normals.unsqueeze(0), neighbor_index=neighbor_index.unsqueeze(0))

    # save embedding
    features = features.detach().cpu()
    output_file_1 = Path(output_directory, f"embedding_{processed_file_1.stem}.npz")
    np.savetxt(str(output_file_1), features)

    # get out embedding
    target_feature_vector_1 = np.loadtxt(str(output_file_1))

    # process second model
    processed_file_2 = process_model(mesh_simplified_path_2, output_directory=output_directory)
    centers, corners, normals, neighbor_index = prepare_one_item(str(processed_file_2))

    centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
    corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
    normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
    neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))

    _, features = model(centers=centers.unsqueeze(0), corners=corners.unsqueeze(0),
                        normals=normals.unsqueeze(0), neighbor_index=neighbor_index.unsqueeze(0))

    # save embedding
    features = features.detach().cpu()
    output_file_2 = Path(output_directory, f"embedding_{processed_file_2.stem}.npz")
    np.savetxt(str(output_file_2), features)

    # get out embedding
    target_feature_vector_2 = np.loadtxt(str(output_file_2))

    diff = 1 - spatial.distance.cosine(target_feature_vector_1, target_feature_vector_2)

    await message.answer(f"Похожесть моделей: {round(diff, 5)}")
