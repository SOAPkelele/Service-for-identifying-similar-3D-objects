import os

import numpy as np
import torch
import torch.utils.data as data


class DatasetLoader(data.Dataset):
    def __init__(self, data_root, max_faces: int = 1024):
        self.root = data_root
        self.max_faces = max_faces

        self.data = []
        for filename in os.listdir(self.root):
            if filename.endswith(".npz"):
                self.data.append(os.path.join(self.root, filename))

    def __getitem__(self, i):
        path = self.data[i]
        data = np.load(path)
        face = data["face"]
        neighbor_index = data["neighbor_index"]

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index, path

    def __len__(self):
        return len(self.data)


def prepare_one_item(path, max_faces: int = 1024):
    data = np.load(path)
    face = data["face"]
    neighbor_index = data["neighbor_index"]

    # fill for n < max_faces with randomly picked faces
    num_point = len(face)
    if num_point < max_faces:
        fill_face = []
        fill_neighbor_index = []
        for i in range(max_faces - num_point):
            index = np.random.randint(0, num_point)
            fill_face.append(face[index])
            fill_neighbor_index.append(neighbor_index[index])
        face = np.concatenate((face, np.array(fill_face)))
        neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

    # to tensor
    face = torch.from_numpy(face).float()
    neighbor_index = torch.from_numpy(neighbor_index).long()

    # reorganize
    face = face.permute(1, 0).contiguous()
    centers, corners, normals = face[:3], face[3:12], face[12:]
    corners = corners - torch.cat([centers, centers, centers], 0)

    return centers, corners, normals, neighbor_index
