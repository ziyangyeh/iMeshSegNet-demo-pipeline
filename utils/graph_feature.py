import numpy as np


def topk(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[topk_index_sort, row_index]
        topk_index_sort = topk_index[0:K, :][topk_index_sort, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:, 0:K][column_index, topk_index_sort]
    return topk_data_sort, topk_index_sort

def knn_cpu(x, k):
    k2 = k * 2
    inner = -2 * np.matmul(x.transpose(), x)
    xx = np.sum(x**2, axis=0, keepdims=True)
    pairwise_distance = -xx - inner - xx.transpose()
    idx = topk(pairwise_distance, K=k, axis=1)[1][:, :k2]
    return idx
def get_graph_feature_cpu(x, k=20, idx=None, dim9=False):
    num_points = x.shape[1]
    if idx is None:
        if dim9 == False:
            idx = knn_cpu(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn_cpu(x[:, 6:], k=k)
    idx = idx.flatten()
    num_dims = x.shape[0]
    x = x.transpose()
    feature = x.reshape(num_points, -1)[idx, :]
    feature = feature.reshape(num_points, k, num_dims)
    x = x.reshape(num_points, 1, num_dims).repeat(k, 1)
    feature = np.concatenate([feature - x, x], axis=2).transpose(2, 0, 1)
    return feature