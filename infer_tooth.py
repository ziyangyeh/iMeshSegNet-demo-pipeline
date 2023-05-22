import numpy as np
import open3d as o3d
import pandas as pd
import torch
import vedo

from models import PointNetReg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tooth_patch_size = 3000
tooth_max_keypoints = 6

reg_params = torch.load('PointNetReg.pth', map_location=device)
reg_model = PointNetReg(num_classes=tooth_max_keypoints, with_dropout=False).to(device)
reg_model.load_state_dict(reg_params)
reg_model.eval()

# Keypoints Reg
print("\tKeypoints Reg...")
lst = []
ori = o3d.io.read_triangle_mesh("test.ply")
ori = ori.simplify_quadric_decimation(target_number_of_triangles=3000)
tooth = vedo.utils.open3d2vedo(ori)
tooth_d = tooth.clone()
if tooth_d.ncells > tooth_patch_size:
    tooth_d = tooth_d.decimate(fraction=tooth_patch_size / tooth_d.ncells)
points = tooth_d.points()
mean_cell_centers = tooth_d.center_of_mass()
points[:, 0:3] -= mean_cell_centers[0:3]
ids = np.array(tooth_d.faces())
cells = points[ids].reshape(tooth_d.ncells, 9).astype(dtype="float32")

normals = tooth_d.normals(cells=True)

barycenters = tooth_d.cell_centers()
barycenters -= mean_cell_centers[0:3]

# normalized data
maxs = points.max(axis=0)
mins = points.min(axis=0)
means = points.mean(axis=0)
stds = points.std(axis=0)
nmeans = normals.mean(axis=0)
nstds = normals.std(axis=0)

for i in range(3):
    cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
    cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
    cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
    barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
    normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

X = np.column_stack((cells, barycenters, normals)).transpose().astype(np.float32)
with torch.no_grad():
    patch_prob_output = (
    reg_model(
        x=torch.from_numpy(X).unsqueeze(0).to(device),
    )
    .transpose(2, 1)
    .softmax(dim=-1)
    .cpu()
    .numpy()
        )
result = np.argmax(patch_prob_output[0, :], axis=-1)
print(result)
key_dict = {}
for idx,key in enumerate(result):
    xyz = tooth.cell_centers()[key]
    key_dict[f"key_{idx+1}_x"] = xyz[0]
    key_dict[f"key_{idx+1}_y"] = xyz[1]
    key_dict[f"key_{idx+1}_z"] = xyz[2]
lst.append(key_dict)

df = pd.DataFrame(lst)
df.to_csv("points.csv", index=False)