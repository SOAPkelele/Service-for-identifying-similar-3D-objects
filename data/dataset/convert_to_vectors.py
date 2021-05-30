import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable

from data.models import DatasetLoader
from nets.MeshNet import MeshNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    model = MeshNet(require_fea=True)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(
        torch.load("/home/soapkelele/PycharmProjects/identifying_similar_3d_models/nets/MeshNet/MeshNet_best_37.pkl")
    )
    model.eval()

    data_root = "ModelNet40_processed"
    data_set = DatasetLoader(data_root=data_root)
    data_loader = data.DataLoader(data_set, batch_size=1, num_workers=2)

    output_dir = "embeddings_256"

    files_vectorized = 0
    for i, (centers, corners, normals, neighbor_index, path) in enumerate(data_loader):
        centers = Variable(torch.cuda.FloatTensor(centers.cuda()))
        corners = Variable(torch.cuda.FloatTensor(corners.cuda()))
        normals = Variable(torch.cuda.FloatTensor(normals.cuda()))
        neighbor_index = Variable(torch.cuda.LongTensor(neighbor_index.cuda()))
        path = Path(path[0])

        _, features = model(centers, corners, normals, neighbor_index)
        output_file = Path(output_dir, f"vectorized_{path.stem}.npz")
        print(str(output_file))
        np.savetxt(str(output_file), features.detach().cpu())
        files_vectorized += 1

    print(f"Vectorization completed: {files_vectorized}/{len(os.listdir(data_root))}")
