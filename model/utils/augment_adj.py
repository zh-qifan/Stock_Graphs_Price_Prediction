import torch

def augment_adj(A, window_size):
    # remove self-loop
    for i in range(A.shape[0]):
        A[i, i] = 0
    # get unit mat
    unit_mat = torch.zeros(window_size, window_size).to(A.device)
    for i in range(window_size):
        unit_mat[i, window_size - 1] = 1
    # augment adj
    A_aug = torch.kron(A, unit_mat)
    edge_index = torch.where(A_aug)
    return torch.cat(edge_index, dim=0).reshape(2, -1)