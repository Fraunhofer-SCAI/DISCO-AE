import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors, KDTree

# # plots
import matplotlib.pyplot as plt

import importlib
meshplot_spec = importlib.util.find_spec("meshplot")
meshplot_found = meshplot_spec is not None
if meshplot_found:
    import meshplot as mp
import matplotlib
from matplotlib import cm

import trimesh
from PIL import Image


def knn_query(X, Y, k=1, return_distance=False, use_scipy=False, dual_tree=False, n_jobs=1):

    if use_scipy:
        tree = scipy.spatial.KDTree(X, leafsize=40)
        dists, matches = tree.query(Y, k=k, workers=-1)
    else:
        # if n_jobs == 1:
        #     tree = KDTree(X, leaf_size=40)
        #     dists, matches = tree.query(Y, k=k, return_distance=True)
        # else:
        tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
        tree.fit(X)
        dists, matches = tree.kneighbors(Y)
        if k == 1:
            dists = dists.squeeze()
            matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches


def knn_query_normals(X, Y, normals1, normals2, k_base=30, return_distance=False, n_jobs=1, verbose=False):
    """
    Compute a NN query ensuring normal consistency.
    k_base determines the number of neighbors first computed for faster computation.
    """
    final_matches = np.zeros(Y.shape[0], dtype=int)
    final_dists = np.zeros(Y.shape[0])

    # FIRST DO KNN SEARCH HOPING TO OBTAIN FAST
    # tree = KDTree(X)  # Tree on (n1,)
    # dists, matches = tree.query(Y, k=k_base, return_distance=True)  # (n2,k), (n2,k)

    dists, matches = knn_query(X, Y, k=k_base, return_distance=True, n_jobs=n_jobs)

    # Check for which vertices the solution is already computed
    isvalid = np.einsum('nkp,np->nk', normals1[matches], normals2) > 0  # (n2, k)
    valid_row = isvalid.sum(1) > 0

    valid_inds = valid_row.nonzero()[0]  # (n',)
    invalid_inds = (~valid_row).nonzero()[0]  # (n2-n')

    if verbose:
        print(f'{valid_inds.size} direct matches and {invalid_inds.size} specific indices')

    # Fill the known values
    final_matches[valid_inds] = matches[(valid_inds, isvalid[valid_inds].argmax(axis=1))]
    if return_distance:
        final_dists[valid_inds] = dists[(valid_inds, isvalid[valid_inds].argmax(axis=1))]

    # Individually check other indices
    n_other = invalid_inds.size
    myit = range(n_other)
    for inv_ind in myit:
        vert_ind = invalid_inds[inv_ind]
        possible_inds = np.nonzero(normals1 @ normals2[vert_ind] > 0)[0]

        if len(possible_inds) == 0:
            final_matches[vert_ind] = matches[vert_ind, 0]
            final_dists[vert_ind] = dists[vert_ind, 0]
            continue

        tree = KDTree(X[possible_inds])
        temp_dist, temp_match_red = tree.query(Y[None, vert_ind], k=1, return_distance=True)

        final_matches[vert_ind] = possible_inds[temp_match_red.item()]
        final_dists[vert_ind] = temp_dist.item()

    if return_distance:
        return final_dists, final_matches
    return final_matches


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def get_rgba(cmap, cmapcolor, vmin, vmax):
    ColorHelper = MplColorHelper(cmapcolor, vmin, vmax)
    rgba = ColorHelper.get_rgb(cmap)
    return rgba[:, :3]



if meshplot_found:
    def double_plot(myMesh1, myMesh2, cmap1=None, cmap2=None, value1=None, value2=None, cmapcolor = 'Blues'):
        if value1 is not None:
            cmap1 = get_rgba(value1, cmapcolor = cmapcolor, vmin=value1.min(), vmax=value1.max())
        if value2 is not None:
            cmap2 = get_rgba(value2, cmapcolor = cmapcolor, vmin=value2.min(), vmax=value2.max())
        d = mp.subplot(myMesh1.vertlist, myMesh1.facelist, c=cmap1, s=[2, 2, 0])
        mp.subplot(myMesh2.vertlist, myMesh2.facelist, c=cmap2, s=[2, 2, 1], data=d)

        def plot_mesh(myMesh, cmap=None, value=None, cmapcolor='Blues'):
            if value is not None:
                cmap = get_rgba(value, cmapcolor=cmapcolor, vmin=value.min(), vmax=value.max())
            mp.plot(myMesh.vertlist, myMesh.facelist, c=cmap)


# def triple_plot(myMesh1, myMesh2, myMesh3, cmap1=None, cmap2=None, cmap3=None, value1=None, value2=None, value3=None, cmapcolor = 'Blues'):
#     if value1 is not None:
#         cmap1 = get_rgba(value1, cmapcolor = cmapcolor, vmin=value1.min(), vmax=value1.max())
#     if value2 is not None:
#         cmap2 = get_rgba(value2, cmapcolor = cmapcolor, vmin=value2.min(), vmax=value2.max())
#     if value3 is not None:
#         cmap3 = get_rgba(value3, cmapcolor = cmapcolor, vmin=value3.min(), vmax=value3.max())
#     d = mp.subplot(myMesh1.vertlist, myMesh1.facelist, c=cmap1, s=[3, 3, 0])
#     mp.subplot(myMesh2.vertlist, myMesh2.facelist, c=cmap2, s=[3, 3, 1], data=d)
#     mp.subplot(myMesh3.vertlist, myMesh3.facelist, c=cmap3, s=[3, 3, 2], data=d)


# # texture transfer

def get_uv(vertices, axes=(0, 1), repeat=1):
    vt = vertices[:, axes]  # (V, 2)
    vt = vt - np.amin(vt, axis=0, keepdims=True)
    vt = repeat * vt / np.amax(vt)
    return vt


def put_uv_map(texture_im, uv, mesh, disp=np.array([0., 0., 0.])):
    im = Image.open(texture_im)
    material = trimesh.visual.texture.SimpleMaterial(image=im)
    uv_visual = trimesh.visual.TextureVisuals(uv=uv, image=im, material=material)
    uved_mesh = trimesh.Trimesh(vertices=scale_to_unit_sphere(mesh.vertices) + disp,
                                faces=mesh.faces,
                                visual=uv_visual, validate=True, process=False)
    return uved_mesh


def v_from_m(m):
    return np.array(trimesh.load(m, process=False).vertices)


def scale_to_unit_sphere(points, center=None, buffer=1.):
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1))) * buffer
    points = points / scale
    return np.asarray(points)


def plot_texture_transfer(m1, m2, pred_map, texture_im):
    if isinstance(m1, str):
        m1 = trimesh.load(m1, process=False)
    if isinstance(m2, str):
        m2 = trimesh.load(m2, process=False)
    tar_uv = get_uv(v_from_m(m2))
    src_uv = np.array([[0, 0]] * len(m1.vertices))
    # for src_ind, tar_ind in enumerate(pred_map):
    src_uv = tar_uv[pred_map]
    disp = np.array([1., 0., 0.])  # TO not superimpose
    uved_src = put_uv_map(texture_im, src_uv, m1)
    uved_tar = put_uv_map(texture_im, tar_uv, m2, disp=disp)

    return trimesh.Scene(geometry=[uved_src, uved_tar])
