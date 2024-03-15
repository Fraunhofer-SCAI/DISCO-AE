import random
import numpy as np
from sklearn.neighbors import KDTree
import scipy.linalg
from scipy.spatial import cKDTree
import torch
import torch.nn as nn


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum((a - b) ** 2, axis=(1, 2))
        return torch.mean(loss)


def farthest_point_sample(xyz, ratio):
    xyz = xyz.t().unsqueeze(0)
    npoint = int(ratio * xyz.shape[1])
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids[0]


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def normalize_area_scale(verts, faces):
    coords = verts[faces]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    face_areas = torch.norm(torch.cross(vec_A, vec_B, dim=-1), dim=1) * 0.5
    total_area = torch.sum(face_areas)

    scale = (1 / torch.sqrt(total_area))
    verts = verts * scale

    # center
    verts = verts - verts.mean(dim=-2, keepdim=True)

    return verts


def get_mask(evals1, evals2, gamma=0.5, device="cpu"):
    scaling_factor = max(torch.max(evals1), torch.max(evals2))
    evals1, evals2 = evals1.to(device) / scaling_factor, evals2.to(device) / scaling_factor
    evals_gamma1, evals_gamma2 = (evals1 ** gamma)[None, :], (evals2 ** gamma)[:, None]

    M_re = evals_gamma2 / (evals_gamma2.square() + 1) - evals_gamma1 / (evals_gamma1.square() + 1)
    M_im = 1 / (evals_gamma2.square() + 1) - 1 / (evals_gamma1.square() + 1)
    return M_re.square() + M_im.square()


def euler_angles_to_rotation_matrix(theta):
    R_x = torch.tensor([[1, 0, 0], [0, torch.cos(theta[0]), -torch.sin(theta[0])], [0, torch.sin(theta[0]), torch.cos(theta[0])]])
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])], [0, 1, 0], [-torch.sin(theta[1]), 0, torch.cos(theta[1])]])
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0], [torch.sin(theta[2]), torch.cos(theta[2]), 0], [0, 0, 1]])

    matrices = [R_x, R_y, R_z]

    R = torch.mm(matrices[2], torch.mm(matrices[1], matrices[0]))
    return R


def get_random_rotation(x, y, z):
    thetas = torch.zeros(3, dtype=torch.float)
    degree_angles = [x, y, z]
    for axis_ind, deg_angle in enumerate(degree_angles):
        rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
        rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
        thetas[axis_ind] = rand_radian_angle

    return euler_angles_to_rotation_matrix(thetas)


def data_augmentation(verts, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    # random rotation
    rotation_matrix = get_random_rotation(rot_x, rot_y, rot_z).to(verts.device)
    verts = verts @ rotation_matrix.T

    # random noise
    noise = std * torch.randn(verts.shape).to(verts.device)
    noise = noise.clamp(-noise_clip, noise_clip)
    verts += noise

    # random scaling
    scales = [scale_min, scale_max]
    scale = scales[0] + torch.rand((3,)) * (scales[1] - scales[0])
    verts = verts * scale.to(verts.device)

    return verts


def augment_batch(data, rot_x=0, rot_y=90, rot_z=0, std=0.01, noise_clip=0.05, scale_min=0.9, scale_max=1.1):
    data["shape1"]["xyz"] = data_augmentation(data["shape1"]["xyz"], rot_x, rot_y, rot_z, std, noise_clip, scale_min, scale_max)
    data["shape2"]["xyz"] = data_augmentation(data["shape2"]["xyz"], rot_x, rot_y, rot_z, std, noise_clip, scale_min, scale_max)

    return data


ANN = False


class KNNSearch(object):
    DTYPE = np.float32
    NJOBS = 8

    def __init__(self, data):
        self.data = np.asarray(data, dtype=self.DTYPE)
        self.kdtree = cKDTree(self.data)

    def query(self, kpts, k, return_dists=False):
        kpts = np.asarray(kpts, dtype=self.DTYPE)
        nndists, nnindices = self.kdtree.query(kpts, k=k, workers=self.NJOBS)
        if return_dists:
            return nnindices, nndists
        else:
            return nnindices

    def query_ball(self, kpt, radius):
        kpt = np.asarray(kpt, dtype=self.DTYPE)
        assert kpt.ndim == 1
        nnindices = self.kdtree.query_ball_point(kpt, radius, workers=self.NJOBS)
        return nnindices


def FM_to_p2p(FM, eigvects1, eigvects2, use_ANN=False):
    if use_ANN and not ANN:
        raise ValueError('Please install pydescent to achieve Approximate Nearest Neighbor')

    k2, k1 = FM.shape

    assert k1 <= eigvects1.shape[1], \
        f'At least {k1} should be provided, here only {eigvects1.shape[1]} are given'
    assert k2 <= eigvects2.shape[1], \
        f'At least {k2} should be provided, here only {eigvects2.shape[1]} are given'

    tree = KNNSearch(eigvects1[:, :k1] @ FM.T)
    matches = tree.query(eigvects2[:, :k2], k=1).flatten()

    return matches


def p2p_to_FM(p2p, eigvects1, eigvects2, A2=None):
    if A2 is not None:
        if A2.shape[0] != eigvects2.shape[0]:
            raise ValueError("Can't compute pseudo inverse with subsampled eigenvectors")

        if len(A2.shape) == 1:
            return eigvects2.T @ (A2[:, None] * eigvects1[p2p, :])

        return eigvects2.T @ A2 @ eigvects1[p2p, :]

    return scipy.linalg.lstsq(eigvects2, eigvects1[p2p, :])[0]


def zoomout_iteration(eigvects1, eigvects2, FM, step=1, A2=None, use_ANN=False):
    k2, k1 = FM.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p = FM_to_p2p(FM, eigvects1, eigvects2, use_ANN=use_ANN)
    FM_zo = p2p_to_FM(p2p, eigvects1[:, :new_k1], eigvects2[:, :new_k2], A2=A2)

    return FM_zo


def zoomout_refine(eigvects1,
                   eigvects2,
                   FM,
                   nit=10,
                   step=1,
                   A2=None,
                   subsample=None,
                   use_ANN=False,
                   return_p2p=False,
                   verbose=False):
    k2_0, k1_0 = FM.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert k1_0 + nit * step1 <= eigvects1.shape[1], \
        f"Not enough eigenvectors on source : \
        {k1_0 + nit*step1} are needed when {eigvects1.shape[1]} are provided"
    assert k2_0 + nit * step2 <= eigvects2.shape[1], \
        f"Not enough eigenvectors on target : \
        {k2_0 + nit*step2} are needed when {eigvects2.shape[1]} are provided"

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_zo = FM.copy()

    ANN_adventage = False
    iterable = range(nit)
    for it in iterable:
        ANN_adventage = use_ANN and (FM_zo.shape[0] > 90) and (FM_zo.shape[1] > 90)

        if use_subsample:
            FM_zo = zoomout_iteration(eigvects1[sub1],
                                      eigvects2[sub2],
                                      FM_zo,
                                      A2=None,
                                      step=step,
                                      use_ANN=ANN_adventage)

        else:
            FM_zo = zoomout_iteration(eigvects1,
                                      eigvects2,
                                      FM_zo,
                                      A2=A2,
                                      step=step,
                                      use_ANN=ANN_adventage)

    if return_p2p:
        p2p_zo = FM_to_p2p(FM_zo, eigvects1, eigvects2, use_ANN=False)
        return FM_zo, p2p_zo

    return FM_zo


def norm(x, highdim=False):
    return torch.norm(x, dim=len(x.shape) - 1)


def toNP(x):
    return x.detach().to(torch.device('cpu')).numpy()


def find_knn(points_source, points_target, k, largest=False, omit_diagonal=False, method='brute'):

    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError("omit_diagonal can only be used when source and target are same shape")

    if method != 'cpu_kd' and points_source.shape[0] * points_target.shape[0] > 1e8:
        method = 'cpu_kd'
        print("switching to cpu_kd knn")

    if method == 'brute':

        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(-1, points_target.shape[0], -1)
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(points_source.shape[0], -1, -1)

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float('inf')

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result

    elif method == 'cpu_kd':

        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = toNP(points_source)
        points_target_np = toNP(points_target)

        # Build the tree
        kd_tree = KDTree(points_target_np)

        k_search = k + 1 if omit_diagonal else k
        _, neighbors = kd_tree.query(points_source_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape((neighbors.shape[0], neighbors.shape[1] - 1))

        inds = torch.tensor(neighbors, device=points_source.device, dtype=torch.int64)
        dists = norm(points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds])

        return dists, inds

    else:
        raise ValueError("unrecognized method")
