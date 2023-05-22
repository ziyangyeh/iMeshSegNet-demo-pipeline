import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import vedo
from pygco import cut_from_graph
from scipy.spatial import distance_matrix

from models import MeshSegNet, PointNetReg
from utils import get_graph_feature_cpu, get_tooth

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
jaw_patch_size = 7000
tooth_patch_size = 3000
jaw_num_classes = 15
tooth_max_keypoints = 6
# svm = "cuml"
svm = "sklearn"

seg_params = torch.load("cont_lower_fold_6.tar", map_location=device)['model_state_dict']
seg_model = MeshSegNet(num_classes=jaw_num_classes, with_dropout=False).to(device)
seg_model.load_state_dict(seg_params)
seg_model.eval()

reg_params = torch.load('PointNetReg.pth', map_location=device)
reg_model = PointNetReg(num_classes=tooth_max_keypoints, with_dropout=False).to(device)
reg_model.load_state_dict(reg_params)
reg_model.eval()

mesh = vedo.Mesh("test_input.vtp")
# pre-processing: downsampling
print('\tDownsampling...')
target_num = 10000
ratio = target_num/mesh.ncells # calculate ratio
mesh_d = mesh.clone()
mesh_d.decimate(fraction=ratio)
predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

# move mesh to origin
print('\tPredicting...')
points = mesh_d.points()
mean_cell_centers = mesh_d.center_of_mass()
points[:, 0:3] -= mean_cell_centers[0:3]

ids = np.array(mesh_d.faces())
cells = points[ids].reshape(mesh_d.ncells, 9).astype(dtype='float32')

# customized normal calculation; the vtk/vedo build-in function will change number of points
mesh_d.compute_normals()
normals = mesh_d.celldata['Normals']

# move mesh to origin
barycenters = mesh_d.cell_centers() # don't need to copy
barycenters -= mean_cell_centers[0:3]

#normalized data
maxs = points.max(axis=0)
mins = points.min(axis=0)
means = points.mean(axis=0)
stds = points.std(axis=0)
nmeans = normals.mean(axis=0)
nstds = normals.std(axis=0)

for i in range(3):
    cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
    cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
    cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
    barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
    normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

X = np.column_stack((cells, barycenters, normals))

# computing A_S and A_L
A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
D = distance_matrix(X[:, 9:12], X[:, 9:12])
A_S[D<0.1] = 1.0
A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

A_L[D<0.2] = 1.0
A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

# numpy -> torch.tensor
X = X.transpose(1, 0)
X = X.reshape([1, X.shape[0], X.shape[1]])
X = torch.from_numpy(X).to(device, dtype=torch.float)
A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

with torch.no_grad():
    tensor_prob_output = seg_model(X, A_S, A_L).to(device, dtype=torch.float)
patch_prob_output = tensor_prob_output.cpu().numpy()

for i_label in range(jaw_num_classes):
    predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

# output downsampled predicted labels
mesh2 = mesh_d.clone()
mesh2.celldata['Label'] = predicted_labels_d
vedo.write(mesh2, "test_output_sim.vtp")

# refinement
print('\tRefining by pygco...')
round_factor = 100
patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

# unaries
unaries = -round_factor * np.log10(patch_prob_output)
unaries = unaries.astype(np.int32)
unaries = unaries.reshape(-1, jaw_num_classes)

# parawise
pairwise = (1 - np.eye(jaw_num_classes, dtype=np.int32))

#edges
normals = mesh_d.celldata['Normals'].copy() # need to copy, they use the same memory address
barycenters = mesh_d.cell_centers() # don't need to copy
cell_ids = np.asarray(mesh_d.faces())

lambda_c = 30
edges = np.empty([1, 3], order='C')
for i_node in range(cells.shape[0]):
    # Find neighbors
    nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
    nei_id = np.where(nei==2)
    for i_nei in nei_id[0][:]:
        if i_node < i_nei:
            cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
            if cos_theta >= 1.0:
                cos_theta = 0.9999
            theta = np.arccos(cos_theta)
            phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
            if theta > np.pi/2.0:
                edges = np.concatenate((edges, np.array([i_node, i_nei, -np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
            else:
                beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*np.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
edges = np.delete(edges, 0, 0)
edges[:, 2] *= lambda_c*round_factor
edges = edges.astype(np.int32)

refine_labels = cut_from_graph(edges, unaries, pairwise)
refine_labels = refine_labels.reshape([-1, 1])

# output refined result
mesh3 = mesh_d.clone()
mesh3.celldata['Label'] = refine_labels
vedo.write(mesh3, "test_output_up.vtp")

with torch.no_grad():
    patch_prob_output = (
    seg_model(
        X.to(device),
        A_S.to(device),
        A_L.to(device),
    )
    .transpose(2, 1)
    .softmax(dim=-1)
    .cpu()
    .numpy()
        )
patch_prob_output = torch.from_numpy(patch_prob_output).transpose(2,1).numpy()
predicted_labels = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
for i_label in range(jaw_num_classes):
    predicted_labels[
        np.argmax(patch_prob_output[0, :], axis=-1) == i_label
    ] = i_label

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