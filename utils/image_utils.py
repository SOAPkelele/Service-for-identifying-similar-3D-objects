from pathlib import Path

import numpy as np
import open3d
import pymesh


def render_model_preview(mesh_path: Path, save_directory: Path, visualizer: open3d.visualization.Visualizer):
    mesh = open3d.io.read_triangle_mesh(str(mesh_path))

    # compute vertex normals to display proper lightning
    mesh.compute_vertex_normals()

    # rotate mesh as it is always displayed from the top view
    # also rotate slightly to see the side of the model
    mesh.rotate(R=mesh.get_rotation_matrix_from_xyz((- 75 * np.pi / 180, 0, 0)))
    mesh.rotate(R=mesh.get_rotation_matrix_from_xyz((0, - np.pi / 6, 0)))
    mesh.rotate(R=mesh.get_rotation_matrix_from_xyz((0, 0, - 15 * np.pi / 180)))

    # add mesh to rendered and save a picture
    visualizer.add_geometry(mesh)
    visualizer.poll_events()

    filename = str(save_directory / f"{mesh_path.stem}_preview.jpg")

    visualizer.capture_screen_image(filename=filename, do_render=True)
    visualizer.remove_geometry(mesh)

    return filename


def simplify_model(mesh_path: Path, save_directory: Path, faces: int = 1024, add_postfix: bool = False):
    """Open mesh, reduce number of faces, clean up and save to file"""
    mesh = open3d.io.read_triangle_mesh(str(mesh_path))
    print(f"Mesh before simplification: {mesh}")
    mesh_simplified: open3d.geometry.TriangleMesh = mesh.simplify_quadric_decimation(target_number_of_triangles=faces)
    mesh_simplified.remove_unreferenced_vertices()
    mesh_simplified.remove_duplicated_vertices()
    print(f"Mesh after simplification: {mesh_simplified}")
    if add_postfix:
        output_file = Path(save_directory, f"{mesh_path.stem}_simplified{mesh_path.suffix}")
    else:
        output_file = save_directory / mesh_path.name
    print(output_file)
    open3d.io.write_triangle_mesh(str(output_file), mesh=mesh_simplified)
    return output_file


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            try:
                face.remove(vf1)
                face.remove(vf2)
            except:
                print("Error removing")
            return i

    return except_face


def process_model(mesh_simplified: Path, output_directory: Path):
    # load mesh
    mesh = pymesh.load_mesh(str(mesh_simplified))

    # get elements
    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    # move to center
    center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
    vertices -= center

    # normalize
    max_len = np.max(vertices[:, 0] ** 2 + vertices[:, 1] ** 2 + vertices[:, 2] ** 2)
    vertices /= np.sqrt(max_len)

    # get normal vector
    mesh = pymesh.form_mesh(vertices, faces)
    mesh.add_attribute('face_normal')
    face_normal = mesh.get_face_attribute('face_normal')

    # get neighbors
    faces_contain_this_vertex = []
    for i in range(len(vertices)):
        faces_contain_this_vertex.append(set([]))

    centers = []
    corners = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        x1, y1, z1 = vertices[v1]
        x2, y2, z2 = vertices[v2]
        x3, y3, z3 = vertices[v3]
        centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
        corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
        faces_contain_this_vertex[v1].add(i)
        faces_contain_this_vertex[v2].add(i)
        faces_contain_this_vertex[v3].add(i)

    neighbors = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
        n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
        n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
        neighbors.append([n1, n2, n3])

    centers = np.array(centers)
    corners = np.array(corners)
    faces = np.concatenate([centers, corners, face_normal], axis=1)
    neighbors = np.array(neighbors)

    file_out = Path(output_directory, mesh_simplified.stem + ".npz")
    np.savez(str(file_out), face=faces, neighbor_index=neighbors)
    return file_out
