import numpy as np


def get_tooth(mesh, label, index):
    clone = mesh.clone()
    cell_idx = np.where(label != index)[0]
    clone.delete_cells(cell_idx)
    return clone