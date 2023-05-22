import numpy as np
import pandas as pd
import torch
import vedo
from pygco import cut_from_graph

from models import PointNetReg, iMeshSegNet
from utils import get_graph_feature_cpu, get_tooth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
jaw_patch_size = 7000
tooth_patch_size = 3000
jaw_num_classes = 15
tooth_max_keypoints = 6
# svm = "cuml"
svm = "sklearn"

seg_params = torch.load('iMeshSegNet.pth', map_location=device)
seg_model = iMeshSegNet(num_classes=jaw_num_classes, with_dropout=False).to(device)
seg_model.load_state_dict(seg_params)
seg_model.eval()

reg_params = torch.load('PointNetReg.pth', map_location=device)
reg_model = PointNetReg(num_classes=tooth_max_keypoints, with_dropout=False).to(device)
reg_model.load_state_dict(reg_params)
reg_model.eval()

mesh = vedo.Mesh("test_input.vtp")
mesh.compute_normals()

mesh_d = mesh.clone()
if mesh_d.ncells > jaw_patch_size:
    mesh_d = mesh_d.decimate(fraction=jaw_patch_size / mesh_d.ncells)
points = mesh_d.points()
mean_cell_centers = mesh_d.center_of_mass()
points[:, 0:3] -= mean_cell_centers[0:3]
ids = np.array(mesh_d.faces())
cells = points[ids].reshape(mesh_d.ncells, 9).astype(dtype="float32")

normals = mesh_d.normals(cells=True)

barycenters = mesh_d.cell_centers()
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

KG_6 = get_graph_feature_cpu(X[9:12, :], k=6)

KG_12 = get_graph_feature_cpu(X[9:12, :], k=12)

with torch.no_grad():
    patch_prob_output = (
    seg_model(
        x=torch.from_numpy(X).unsqueeze(0).to(device),
        kg_6=torch.from_numpy(KG_6).unsqueeze(0).to(device),
        kg_12=torch.from_numpy(KG_12).unsqueeze(0).to(device),
    )
    .transpose(2, 1)
    .softmax(dim=-1)
    .cpu()
    .numpy()
        )
predicted_labels = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
for i_label in range(jaw_num_classes):
    predicted_labels[
        np.argmax(patch_prob_output[0, :], axis=-1) == i_label
    ] = i_label

# refinement
print("\tRefining by pygco...")
round_factor = 100
patch_prob_output[patch_prob_output < 1.0e-6] = 1.0e-6

# unaries
unaries = -round_factor * np.log10(patch_prob_output)
unaries = unaries.astype(np.int32)
unaries = unaries.reshape(-1, jaw_num_classes)

# parawise
pairwise = 1 - np.eye(jaw_num_classes, dtype=np.int32)

lambda_c = 30
edges = np.empty([1, 3], order="C")
cell_ids = np.asarray(mesh_d.faces())
for i_node in range(cells.shape[0]):
    # Find neighbors
    nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
    nei_id = np.where(nei == 2)
    for i_nei in nei_id[0][:]:
        if i_node < i_nei:
            cos_theta = (
                np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])
                / np.linalg.norm(normals[i_node, 0:3])
                / np.linalg.norm(normals[i_nei, 0:3])
            )
            if cos_theta >= 1.0:
                cos_theta = 0.9999
            theta = np.arccos(cos_theta)
            phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
            if theta > np.pi / 2.0:
                edges = np.concatenate(
                    (
                        edges,
                        np.array(
                            [i_node, i_nei, -np.log10(theta / np.pi) * phi]
                        ).reshape(1, 3),
                    ),
                    axis=0,
                )
            else:
                beta = 1 + np.linalg.norm(
                    np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])
                )
                edges = np.concatenate(
                    (
                        edges,
                        np.array(
                            [i_node, i_nei, -beta * np.log10(theta / np.pi) * phi]
                        ).reshape(1, 3),
                    ),
                    axis=0,
                )
edges = np.delete(edges, 0, 0)
edges[:, 2] *= lambda_c * round_factor
edges = edges.astype(np.int32)

refine_labels = cut_from_graph(edges, unaries, pairwise)
refine_labels = refine_labels.reshape([-1, 1])

# output refined result
mesh_refined = mesh_d.clone()
mesh_refined.celldata["Label"] = refine_labels

# upsample
print("\tUpsampling by SVM...")
# train SVM
if svm == "sklearn":
    from sklearn.svm import SVC
elif svm == "cuml":
    from cuml.svm import SVC
clf = SVC(kernel="rbf", gamma="auto")
clf.fit(mesh_refined.cell_centers(), np.ravel(refine_labels))
fine_labels = clf.predict(mesh.cell_centers())
fine_labels = fine_labels.reshape(-1, 1)
mesh.celldata["Label"] = fine_labels
vedo.write(mesh_refined, "test_output.vtp")

# Keypoints Reg
print("\tKeypoints Reg...")
lst = []
for i in np.unique(fine_labels.ravel())[1:]:
    tooth = get_tooth(mesh,fine_labels, i)
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
    key_dict = {}
    for idx,key in enumerate(result):
        xyz = tooth.cell_centers()[key]
        key_dict[f"key_{idx+1}_x"] = xyz[0]
        key_dict[f"key_{idx+1}_y"] = xyz[1]
        key_dict[f"key_{idx+1}_z"] = xyz[2]
    lst.append(key_dict)

df = pd.DataFrame(lst)
df.to_csv("points.csv", index=False)